import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim, expert_type):
        super(ExpertAgent, self).__init__()
        self.expert_type = expert_type
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc3(x))

    @staticmethod
    def create_experts(input_dim, hidden_dim, num_experts):
        experts = nn.ModuleList([
            ExpertAgent(input_dim, hidden_dim, f"Expert {i + 1}")
            for i in range(num_experts)
        ])
        return experts