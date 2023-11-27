import torch


class ModelV1(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelV1, self).__init__()
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = self.fc1(x)
        return x