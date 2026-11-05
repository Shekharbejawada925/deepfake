"""ResNeXt50 spatial encoder and a true temporal LSTM, matching the methodology."""
import torch
from torch import nn
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights


class VideoModel(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V1 if pretrained else None
        cnn = resnext50_32x4d(weights=weights)
        self.model = nn.Sequential(*list(cnn.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(input_size=2048, hidden_size=2048, num_layers=1,
                            bias=True, batch_first=True, bidirectional=False)
        self.relu = nn.LeakyReLU(0.01)
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, 2)

    def encode(self, frames):
        return self.avgpool(self.model(frames)).flatten(1)

    def classify(self, features):
        if features.ndim != 3 or features.shape[-1] != 2048:
            raise ValueError("Features must have shape [batch, time, 2048].")
        sequence, _ = self.lstm(features)
        return self.linear1(self.dp(self.relu(sequence[:, -1, :])))

    def forward(self, x):
        if x.ndim != 5 or x.shape[2:] != (3, 112, 112):
            raise ValueError("Expected [batch, time, 3, 112, 112].")
        batch, length = x.shape[:2]
        features = self.encode(x.reshape(-1, 3, 112, 112))
        return self.classify(features.reshape(batch, length, 2048))


class LegacyVideoModel(VideoModel):
    """Preserve the uploaded app's eval graph for one video at a time.

    The original sequence-first LSTM receives [1, time, feature]. Its sequence
    length is therefore one. Only the selected final frame affects the output.
    This compatibility path does not pretend to repair a trained temporal model.
    """
    def __init__(self):
        super().__init__(pretrained=False)
        self.lstm = nn.LSTM(input_size=2048, hidden_size=2048, num_layers=1,
                            bias=False, batch_first=False, bidirectional=False)

    def classify(self, features):
        if features.ndim != 3 or features.shape[0] != 1 or features.shape[-1] != 2048:
            raise ValueError("Legacy inference accepts one video at a time: [1, time, 2048].")
        sequence, _ = self.lstm(features)
        # The original LeakyReLU was declared but never called.
        # Dropout after the linear layer is inactive in eval mode.
        return self.dp(self.linear1(sequence[:, -1, :]))


def select_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # CPU is the portable default; MPS is available through DEVICE=mps.
    return torch.device("cpu")
