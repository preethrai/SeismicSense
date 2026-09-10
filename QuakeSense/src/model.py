import torch
import torch.nn as nn


class TokenEncoder(nn.Module):

    def __init__(
        self,
        input_len,
        embed_dim=32,
        hidden_dim=64,
        dropout=0.3
    ):

        super().__init__()

        self.conv1 = nn.Conv1d(
            1,
            16,
            kernel_size=5,
            padding=2
        )

        self.conv2 = nn.Conv1d(
            16,
            32,
            kernel_size=5,
            padding=2
        )

        self.pool = nn.MaxPool1d(2)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            dropout
        )

        reduced_len = (
            input_len // 4
        )

        self.fc1 = nn.Linear(
            32 * reduced_len,
            hidden_dim
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            embed_dim
        )


    def forward(self, x):

        x = self.pool(
            self.relu(
                self.conv1(x)
            )
        )

        x = self.pool(
            self.relu(
                self.conv2(x)
            )
        )

        x = x.view(
            x.size(0),
            -1
        )

        x = self.fc1(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return self.relu(x)


# ---------------------------------------------------------------------------
# Step 7: CRNN
# ---------------------------------------------------------------------------

class CRNN(nn.Module):

    def __init__(
        self,
        token_len,
        baseline_dim,
        embed_dim=32,
        hidden_dim=64,
        baseline_hidden=16
    ):

        super().__init__()

        self.encoder = TokenEncoder(
            token_len,
            embed_dim
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # Convert baseline statistics into
        # a small learned representation.

        self.baseline_net = nn.Sequential(

            nn.Linear(
                baseline_dim,
                baseline_hidden
            ),

            nn.ReLU(),

            nn.Dropout(0.2)

        )


        # Combine:
        #
        # LSTM hidden state = 64
        # baseline embedding = 16
        #
        # total = 80

        self.fc_out = nn.Linear(
            hidden_dim + baseline_hidden,
            1
        )


    def forward(
        self,
        x,
        baseline
    ):

        # x:
        # (batch, sequence, 200)

        b, seq_len, token_len = (
            x.shape
        )


        # CNN processes each token

        x = x.view(
            b * seq_len,
            1,
            token_len
        )

        embeds = self.encoder(x)


        # Restore sequence

        embeds = embeds.view(
            b,
            seq_len,
            -1
        )


        # Temporal processing

        out, _ = self.lstm(
            embeds
        )


        # Baseline information

        baseline_embedding = (
            self.baseline_net(
                baseline
            )
        )


        # Repeat baseline for every timestep

        baseline_embedding = (
            baseline_embedding
            .unsqueeze(1)
            .expand(
                -1,
                seq_len,
                -1
            )
        )


        # Combine temporal information
        # with noise baseline.

        combined = torch.cat(
            [
                out,
                baseline_embedding
            ],
            dim=-1
        )


        logits = self.fc_out(
            combined
        ).squeeze(-1)


        return logits
