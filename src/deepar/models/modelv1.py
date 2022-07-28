"""
DeepAR model
"""

import torch.nn as nn
import torch


class DeepARV1(nn.Module):
    def __init__(self, params, num_classes):
        """
        DeepARV1

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

        linear_dim = params.lstm_hidden_dim * params.lstm_layers

        if params.model_type == "basic":
            self.distribution_mu = nn.Linear(linear_dim, 1)
            self.distribution_presigma = nn.Linear(linear_dim, 1)
        elif params.model_type == "output_mlp":
            self.distribution_mu = nn.Sequential(
                nn.Linear(linear_dim, linear_dim // 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(linear_dim // 4, 1)
            )
            self.distribution_presigma = nn.Sequential(
                nn.Linear(linear_dim, linear_dim // 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(linear_dim // 4, 1)
            )
        else:
            raise NotImplementedError

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

    def forward(self, x, idx, hidden, cell):
        """
        Predict mu and sigma of the distribution for z_t.
        Args:
            x: ([1, batch_size, 1+cov_dim]): z_{t-1} + x_t, note that z_0 = 0
            idx ([1, batch_size]): one integer denoting the time series id
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step t-1
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step t-1
        Returns:
            mu ([batch_size]): estimated mean of z_t
            sigma ([batch_size]): estimated standard deviation of z_t
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step t
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step t
        """
        # TODO: do embedding once per window instead of per step, for efficiency?
        idx_embed = self.embedding(idx)
        lstm_input = torch.cat((x, idx_embed), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        # use hidden from all lstm layers to compute mu and sigma
        # hidden_permute: (num_layers, B, hidden_dim) -> (B, hidden_dim, num_layers) -> (B, hidden_dim * num_layers)
        hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
        pre_sigma = self.distribution_presigma(hidden_permute)
        sigma = self.distribution_sigma(pre_sigma)
        mu = self.distribution_mu(hidden_permute)
        return torch.squeeze(mu), torch.squeeze(sigma), hidden, cell

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
