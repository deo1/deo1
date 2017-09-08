import torch
import moduleplus as mp
import torch.nn as nn
import torch.autograd

class TaxiCombinerNet(mp.ModulePlus):
    def __init__(self, input_nodes, learn_rate=0.01, cuda=False, max_output=float("inf")):
        super().__init__()
        
        self.model == nn.Sequential(
            nn.Linear(input_nodes, 1),
            nn.ReLU()
        )

        # initialize weights
        for f in self.model:
            if isinstance(f, torch.nn.modules.linear.Linear):
                nn.init.kaiming_normal(f.weight)
        
        self.max_output = max_output
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate)

        def forward(self, x):
            return torch.clamp(self.model(x), max=self.max_output)

class TaxiNet(mp.ModulePlus):
    def __init__(self, input_nodes, learn_rate=0.01, cuda=False, max_output=float("inf")):
        super().__init__(learn_rate=learn_rate, cuda=cuda)
        
        # TODO: batchnorm layers cause bug with cuda=True
        # TODO: convolutional layers on the coordinates
        self.model = nn.Sequential(
            # Layer 1
            nn.Linear(input_nodes, 50, bias=False), # affine (bias redundant with beta term in batchnorm)
            nn.BatchNorm1d(50),                     # normalize mean/variance
            nn.PReLU(50),                           # adaptive leaky
            
            # Layer 2
            nn.Linear(50, 30, bias=False),          # affine
            nn.BatchNorm1d(30),                     # normalize
            nn.PReLU(30),                           # adaptive leaky

            # Layer 3
            nn.Linear(30, 20, bias=False),          # affine
            nn.BatchNorm1d(20),                     # normalize
            nn.PReLU(20),                           # adaptive leaky

            # Layer 4
            nn.Linear(20, 15, bias=False),          # affine
            nn.BatchNorm1d(15),                     # normalize
            nn.PReLU(15),                           # adaptive leaky

            # Layer 5
            nn.Linear(15, 10, bias=False),          # affine
            nn.BatchNorm1d(10),                     # normalize
            nn.Sigmoid(),
            nn.PReLU(10),                           # adaptive leaky

            # Layer 6
            nn.Linear(10, 1),                       # affine
            nn.ReLU()                               # final output is [0, oo)
        )

        # initialize weights
        for f in self.model:
            if isinstance(f, torch.nn.modules.linear.Linear):
                nn.init.kaiming_normal(f.weight)

        self.max_output = max_output
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate)

    def forward(self, x):
        return torch.clamp(self.model(x), max=self.max_output)

