"""Correct temporal architecture for RE-TRAINING, not the original checkpoint.

Do not load legacy LSTM parameters directly and claim improved accuracy.
"""
from torch import nn
from torchvision.models import resnext50_32x4d

class TemporalDeepfakeModel(nn.Module):
    def __init__(self, num_classes=2, hidden_dim=512):
        super().__init__()
        backbone = resnext50_32x4d(weights=None)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.temporal = nn.LSTM(2048, hidden_dim, batch_first=True,
                                num_layers=1, bidirectional=False)
        self.classifier = nn.Sequential(nn.Dropout(0.4),nn.Linear(hidden_dim,num_classes))

    def forward(self, x):
        b,t,c,h,w=x.shape
        z=self.encoder(x.reshape(b*t,c,h,w)).reshape(b,t,2048)
        z,_=self.temporal(z)
        return self.classifier(z[:,-1])
