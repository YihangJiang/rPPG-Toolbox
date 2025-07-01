import torch
import torch.nn as nn
import torchvision.models as models

class CNNRNNModel(nn.Module):
    def __init__(self, chunk_len, hidden_dim=128, lstm_layers=2, dropout=0.2):
        super().__init__()

        self.chunk_len = chunk_len

        # CNN: ResNet18 without final pooling and FC
        resnet = models.resnet18(pretrained=False)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])  # B x 512 x H' x W'

        # Pool spatial output to vector
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # → B x 512 x 1 x 1

        # RNN for temporal modeling
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # Output: 1D rPPG waveform
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        Input shape: (N, D, C, H, W)
        Output shape: (N*D, 1)
        """
        x = x[:,:3,:,:]
        N_D, C, H, W = x.shape

        N = N_D // self.chunk_len
        D = self.chunk_len

        x = x.view(N * D, C, H, W)
        feats = self.cnn_backbone(x)                    # (N*D, 512, H', W')
        feats = self.global_pool(feats).squeeze(-1).squeeze(-1)  # (N*D, 512)

        feats = feats.view(N, D, -1)                    # (N, D, 512)
        rnn_out, _ = self.rnn(feats)                    # (N, D, 2*hidden_dim)
        output = self.fc(rnn_out)                       # (N, D, 1)

        return output.view(N * D, 1)                    # (N*D, 1)
