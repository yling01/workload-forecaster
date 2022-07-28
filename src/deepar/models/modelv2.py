"""
DeepAR model
"""

import torch.nn as nn
import torch


class DeepARV2(nn.Module):
    """
    DeepARV2

    DeepARV2 is a variant over DeepARV1. DeepARV1 uses hidden states from all
    layers to predict mu and sigma, which requires manual unrolling of RNN
    forwarding to collect all hidden states. DeepARV2 uses only the hidden 
    states of last layer, to avoid manual unrolling and (hopefully) speedup 
    training.
    """

    def __init__(self, params, num_classes):
        """
        DeepAR model that predicts future values of a time-dependent variable 
        based on past values and covariates.
        """
        super().__init__()
        self.params = params

        # num_class: number of timeseries
        self.embedding = nn.Embedding(num_classes, params.embedding_dim)

        self.lstm = nn.LSTM(
            input_size=1 + params.cov_dim + params.embedding_dim,
            hidden_size=params.lstm_hidden_dim,
            num_layers=params.lstm_layers,
            bias=True,
            batch_first=False,
            dropout=params.lstm_dropout,
        )

        self.relu = nn.ReLU()

        self.distribution_mu = nn.Linear(params.lstm_hidden_dim, 1)
        self.distribution_presigma = nn.Linear(params.lstm_hidden_dim, 1)
        self.distribution_sigma = nn.Softplus()
        self._init_weights()

    def _init_weights(self):
        # initialize LSTM forget gate bias to 1 as recommanded by
        # http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.0)

    def forward(self, zxs, series_ids):
        """
        zxs ([T, B, 1+cov_dim])
        series_id ([1, B, 1])
        """
        T, B = zxs.shape[0], zxs.shape[1]

        idx_embed = self.embedding(series_ids).repeat([T, 1, 1])  # (1, B, embed_dim) -> (T, B, embed_dim)
        lstm_input = torch.cat((zxs, idx_embed), dim=2)

        # output: (T, B, hidden_dim)
        output, _ = self.lstm(lstm_input)
        mu = self.distribution_mu(output)
        sigma = self.distribution_sigma(self.distribution_presigma(output))
        return mu.squeeze(), sigma.squeeze()

    def init_hidden(self, input_size):
        """
        Initialize hidden states
        """
        return torch.zeros(self.params.lstm_layers, input_size, self.params.lstm_hidden_dim, device=self.params.device)

    def init_cell(self, input_size):
        """
        Initialize cell states
        """
        return torch.zeros(self.params.lstm_layers, input_size, self.params.lstm_hidden_dim, device=self.params.device)

    def test(self, zxs, vs, series_ids, hidden, cell, sampling):
        """
        sampling: whether using ancestral sampling
        """
        B = zxs.shape[1]
        if sampling:
            samples = torch.zeros(self.params.sample_times, B, self.params.predict_steps, device=self.params.device)
            for j in range(self.params.sample_times):
                decoder_hidden = hidden
                decoder_cell = cell
                for t in range(self.params.predict_steps):
                    mu_de, sigma_de, decoder_hidden, decoder_cell = self(
                        zxs[self.params.predict_start + t].unsqueeze(0), series_ids, decoder_hidden, decoder_cell
                    )
                    gaussian = torch.distributions.normal.Normal(mu_de, sigma_de)
                    pred = gaussian.sample()  # not scaled
                    samples[j, :, t] = pred * vs[:, 0] + vs[:, 1]
                    if t < (self.params.predict_steps - 1):
                        zxs[self.params.predict_start + t + 1, :, 0] = pred

            sample_mu = torch.median(samples, dim=0)[0]
            sample_sigma = samples.std(dim=0)
            return samples, sample_mu, sample_sigma
        else:
            pred_mu = torch.zeros(B, self.params.predict_steps, device=self.params.device)
            pred_sigma = torch.zeros(B, self.params.predict_steps, device=self.params.device)

            h, c = hidden, cell
            for t in range(self.params.predict_steps):
                mu_de, sigma_de, h, c = self(zxs[self.params.predict_start + t].unsqueeze(0), series_ids, h, c)

                pred_mu[:, t] = mu_de * vs[:, 0] + vs[:, 1]
                pred_sigma[:, t] = sigma_de * vs[:, 0]

                if t < (self.params.predict_steps - 1):
                    zxs[self.params.predict_start + t + 1, :, 0] = mu_de

            return pred_mu, pred_sigma
