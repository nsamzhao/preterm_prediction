import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class AllGenesNet(BaseModel):
    def __init__(self, nfeatures = 29459, nclasses = 2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(nfeatures, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(64, nclasses),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.model(x)
        return y

class Net(BaseModel):
    def __init__(self, nfeatures = 100, nclasses = 2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(nfeatures, 50),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(50, 20),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(20, 8),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(8, nclasses),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.model(x)
        return y
