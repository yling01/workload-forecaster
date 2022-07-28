"""
DeepAR model
"""

import torch.nn as nn
import torch
from torchaudio.transforms import FrequencyMasking, TimeMasking


class LockedDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        """
        Args:
            x (:class:`torch.FloatTensor` [batch size, sequence length, rnn hidden size]): Input to
                apply dropout too.
        """
        if not self.training or not self.p:
            return x
        x = x.clone()
        # mask = x.new_empty(1, x.size(1), x.size(2), requires_grad=False).bernoulli_(1 - self.p)
        mask = x.new_empty(x.size(0), 1, x.size(2), requires_grad=False).bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p)
        mask = mask.expand_as(x)
        return x * mask

    def __repr__(self):
        return self.__class__.__name__


class LSTM_with_dropout(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTM_with_dropout, self).__init__()
        self.blstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True, batch_first=True
        )
        self.dropout = LockedDropout(0.15)

    def forward(self, x, hidden, c):
        # X: (B, T, dim)
        x = self.dropout(x)

        out_lstm, (hidden, c) = self.blstm(x, (hidden, c))
        return out_lstm, hidden, c


class Encoder(nn.Module):
    def __init__(self, params, num_classes):
        super(Encoder, self).__init__()
        self.params = params

        self.embedding = nn.Embedding(num_classes, params.cluster_embedding_dim)

        self.initial_lstm = nn.LSTM(
            input_size=1 + params.cov_dim + params.cluster_embedding_dim,
            hidden_size=params.lstm_hidden_dim,
            num_layers=1,
            bias=True,
            bidirectional=True,
            batch_first=True,
            dropout=params.lstm_dropout,
        )

        lstm_modules = []
        for _ in range(self.params.lstm_layers - 1):
            lstm_modules.append(LSTM_with_dropout(params.lstm_hidden_dim * 2, params.lstm_hidden_dim))
        self.LSTMs = nn.ModuleList(lstm_modules)

        linear_dim = params.lstm_hidden_dim * 2

        self.key_network = nn.Linear(linear_dim, params.key_value_size)
        self.value_network = nn.Linear(linear_dim, params.key_value_size)
        self.freq_transform = FrequencyMasking(8)

    def forward(self, zxs, series_ids, mode="train"):
        """
        zxs = (B, T, 1+cov_dim)
        series_ids = (B, )
        """
        B, T = zxs.shape[0], self.params.predict_start
        zxs = zxs[:, :T, :]
        series_ids = series_ids.unsqueeze(1)  # (B, 1)
        idx_embed = self.embedding(series_ids)  # (B, 1, embedding_dim)
        lstm_input = torch.cat((zxs, idx_embed.repeat(1, T, 1)), dim=2)  # (B, T, 1+cov_dim+embedding_dim)

        if mode == "train":
            lstm_input = lstm_input.permute((0, 2, 1))  # (B, T, 1+cov_dim+embedding_dim)
            lstm_input = self.freq_transform(lstm_input)
            lstm_input = lstm_input.permute((0, 2, 1))  # (B, T, 1+cov_dim+embedding_dim)

        hidden, cell = self.init_hidden(B), self.init_cell(B)
        output, (hidden, cell) = self.initial_lstm(lstm_input, (hidden, cell))  # (B, T, hidden*2)
        for lstm in self.LSTMs:
            output, hidden, cell = lstm(output, hidden, cell)  # (B, T, hidden*2)

        key = self.key_network(output)  # (B, T, d_k)
        value = self.value_network(output)  # (B, T, d_k)

        return key, value, idx_embed.squeeze(1)  # (B, embedding_dim)

    def init_hidden(self, input_size):
        """
        Initialize hidden states
        """
        return torch.zeros(2, input_size, self.params.lstm_hidden_dim, device=self.params.device)

    def init_cell(self, input_size):
        """
        Initialize cell states
        """
        return torch.zeros(2, input_size, self.params.lstm_hidden_dim, device=self.params.device)


class Decoder(nn.Module):
    """
    As mentioned in a previous recitation, each forward call of decoder deals with just one time step.
    Thus we use LSTMCell instead of LSTM here.
    The output from the last LSTMCell can be used as a query for calculating attention.
    Methods like Gumble noise and teacher forcing can also be incorporated for improving the performance.
    """

    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params

        self.lstm1 = nn.LSTMCell(
            input_size=1 + params.cov_dim + params.cluster_embedding_dim + self.params.key_value_size,
            hidden_size=self.params.lstm_hidden_dim,
        )
        self.lstm2 = nn.LSTMCell(input_size=self.params.lstm_hidden_dim, hidden_size=self.params.key_value_size)

        self.attention = nn.MultiheadAttention(
            self.params.key_value_size,
            self.params.num_heads,
            self.params.attention_dropout,
            device=self.params.device,
            batch_first=True,
        )

        # TODO: Add gate

        self.dropout = nn.Dropout(0.1)

        linear_dim = self.params.key_value_size + self.params.key_value_size

        # TODO: Add ReLU()?
        self.distribution_mu = nn.Sequential(nn.Linear(linear_dim, 1))
        self.distribution_presigma = nn.Sequential(nn.Linear(linear_dim, 1))
        self.distribution_sigma = nn.Softplus()

        # Weight tying
        # self.character_prob.weight = self.embedding.weight

    def forward(self, zxs, key, value, idx_embed, mode="train", sampling=False):
        """
        Args:
            zxs: (B, T, 1+cov_dim)
            key :(B, T, d_k) - Output of the Encoder (possibly from the Key projection layer)
            value: (B, T, d_v) - Output of the Encoder (possibly from the Value projection layer)
            idx_embed: (B, embed_dim)
            mode: Train or eval mode for teacher forcing 
        """
        B = key.shape[0]

        pred_mu = []
        pred_sigma = []
        attention_plot = []
        hidden_states = [None, None]
        context = torch.zeros((B, self.params.key_value_size), device=self.params.device)  # (B, d_attn)
        for t in range(self.params.predict_steps):
            x = zxs[:, self.params.predict_start + t, :]  # (B, 1+cov_dim)

            lstm_input = torch.cat((x, idx_embed, context), dim=1)  # (B, 1+cov_dim+embed_dim+d_attn)

            # Compute query vector
            hidden_states[0] = list(self.lstm1(lstm_input, hidden_states[0]))  # (B, lstm_in_dim)
            hidden_states[0][0] = self.dropout(hidden_states[0][0])
            hidden_states[1] = list(self.lstm2(hidden_states[0][0], hidden_states[1]))  # (B, key_value_size)
            hidden_states[1][0] = self.dropout(hidden_states[1][0])
            query = hidden_states[1][0]  # (B, key_value_size)

            # Compute attention from the output of the second LSTM Cell
            context, attention = self.attention(query.unsqueeze(1), key, value, need_weights=True)  # (B, 1, d_attn)
            attention_plot.append(attention[0][0].detach().cpu())  # attention[0][0]: (T, )
            context = context.squeeze(1)  # (B, d_attn)
            output_context = torch.cat([query, context], dim=1)  # (B, d_attn+key_value_size)
            output_context = self.dropout(output_context)

            pre_sigma = self.distribution_presigma(output_context)
            sigma = self.distribution_sigma(pre_sigma)  # (B, 1)
            mu = self.distribution_mu(output_context)  # (B, 1)

            if mode == "test":
                if sampling:
                    gaussian = torch.distributions.normal.Normal(mu, sigma)
                    mu = gaussian.sample()  # (B, 1)
                # Use current prediction as next input
                if t < (self.params.predict_steps - 1):
                    zxs[:, self.params.predict_start + t + 1, 0] = mu.squeeze(1)
            else:
                # TODO: teacher forcing
                pass

            # store predictions
            pred_mu.append(mu)
            pred_sigma.append(sigma)

        pred_mu = torch.cat(pred_mu, dim=1)  # (B, predict_steps)
        pred_sigma = torch.cat(pred_sigma, dim=1)  # (B, predict_steps)
        attentions = torch.stack(attention_plot, dim=1)  # (input_len, pred_len)

        return pred_mu, pred_sigma, attentions


class DeepARSeq2Seq(nn.Module):
    def __init__(self, params, num_classes):
        super(DeepARSeq2Seq, self).__init__()
        self.params = params
        self.encoder = Encoder(params, num_classes)
        self.decoder = Decoder(params)

    def forward(self, zxs, series_ids):
        """
        Args:
            zxs: (B, T, 1+cov+dim)
            series_ids: (B, )
        """
        key, value, idx_embed = self.encoder(zxs, series_ids, mode="train")
        pred_mu, pred_sigma, attentions = self.decoder(zxs, key, value, idx_embed, mode="train", sampling=False)

        return pred_mu, pred_sigma, attentions

    def test(self, zxs, vs, series_ids, sampling):
        """
        zxs: (B, T, 1+cov+dim)
        series_ids: (B, )
        vs: (B, 2)
        sampling: whether using ancestral sampling
        """
        B = zxs.shape[0]
        key, value, idx_embed = self.encoder(zxs, series_ids, mode="test")

        if sampling:
            samples = torch.zeros(self.params.sample_times, B, self.params.predict_steps, device=self.params.device)
            for j in range(self.params.sample_times):
                sample_preds, _, _ = self.decoder(zxs, key, value, idx_embed, mode="test", sampling=True)
                samples[j, :, :] = sample_preds * vs[:, 0].unsqueeze(1) + vs[:, 1].unsqueeze(1)  # (B, 16)

            sample_mu = torch.median(samples, dim=0)[0]  # (B, 16)
            sample_sigma = samples.std(dim=0)  # (B, 16)
            return samples, sample_mu, sample_sigma
        else:
            pred_mu, pred_sigma, _ = self.decoder(zxs, key, value, idx_embed, mode="test", sampling=False)

            # Scale mu and sigma
            pred_mu = pred_mu * vs[:, 0].unsqueeze(1) + vs[:, 1].unsqueeze(1)  # (B, 16)
            pred_sigma = pred_sigma * vs[:, 0].unsqueeze(1) + 1e-5  # (B, 16)

            return None, pred_mu, pred_sigma

