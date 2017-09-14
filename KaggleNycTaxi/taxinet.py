import torch
import moduleplus as mp
import torch.nn as nn
import torch.autograd

class TaxiCombinerNet(mp.ModulePlus):
    def __init__(self, input_nodes, learn_rate=0.01, cuda=False, max_output=float("inf")):
        super().__init__(learn_rate=learn_rate, cuda=cuda)
        
        self.model = nn.Sequential(
            nn.Linear(input_nodes, 1)
        )

        # initialize weights
        for f in self.model:
            if isinstance(f, torch.nn.modules.linear.Linear):
                # start with uniform weighted sum of inputs since this is a
                # regressor of optimal-ish estimates
                nn.init.uniform(f.weight, input_nodes**-1, input_nodes**-1)
        
        self.max_output = max_output
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate)

    def forward(self, x):
        return torch.clamp(self.model(x), max=self.max_output, min=1)

class TaxiNet(mp.ModulePlus):
    def __init__(self, input_nodes, learn_rate=0.01, cuda=False, max_output=float("inf")):
        super().__init__(learn_rate=learn_rate, cuda=cuda)
        
        # TODO: batchnorm layers cause bug with cuda=True
        # TODO: convolutional layers on the coordinates
        self.model = nn.Sequential(
            # Layer 1
            nn.Linear(input_nodes, 100, bias=False), # affine (bias redundant with beta term in batchnorm)
            nn.BatchNorm1d(100),                     # normalize mean/variance
            nn.PReLU(100),                           # adaptive leaky
            
            # Layer 2
            nn.Linear(100, 75, bias=False),         # affine
            nn.BatchNorm1d(75),                     # normalize
            nn.PReLU(75),                           # adaptive leaky

            # Layer 3
            nn.Linear(75, 69, bias=False),          # affine
            nn.BatchNorm1d(69),                     # normalize
            nn.PReLU(69),                           # adaptive leaky

            # Layer 4
            nn.Linear(69, 50, bias=False),          # affine
            nn.BatchNorm1d(50),                     # normalize
            nn.PReLU(50),                           # adaptive leaky

            # Layer 5
            nn.Linear(50, 40, bias=False),          # affine
            nn.BatchNorm1d(40),                     # normalize
            nn.PReLU(40),                           # adaptive leaky

            # Layer 6
            nn.Linear(40, 30, bias=False),          # affine
            nn.BatchNorm1d(30),                     # normalize
            nn.PReLU(30),                           # adaptive leaky

            # Layer 7
            nn.Linear(30, 25, bias=False),          # affine
            nn.BatchNorm1d(25),                     # normalize
            nn.PReLU(25),                           # adaptive leaky

            # Layer 8
            nn.Linear(25, 20, bias=False),          # affine
            nn.BatchNorm1d(20),                     # normalize
            nn.PReLU(20),                           # adaptive leaky

            # Layer 9
            nn.Linear(20, 15, bias=False),          # affine
            nn.BatchNorm1d(15),                     # normalize
            nn.PReLU(15),                           # adaptive leaky

            # Layer 10
            nn.Linear(15, 1),                       # affine
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
        return torch.clamp(self.model(x), max=self.max_output, min=1)