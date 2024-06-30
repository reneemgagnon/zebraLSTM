"""
Striped LSTM Implementation
Copyright (c) 2024 Renee M Gagnon
This script is dual-licensed. See the LICENSE.md file for details.
For non-commercial and research use: CC BY-NC 4.0
For commercial use: Contact renee@freedomfamilyconsulting.ca
"""

import torch
from src.zebra_lstm import StripedLSTM

# Example usage of ZebraLSTM
input_size = 100
hidden_size = 200
num_layers = 2
num_stripes = 4
batch_size = 32
seq_length = 50

model = StripedLSTM(input_size, hidden_size, num_layers, num_stripes).cuda()
input_data = torch.randn(seq_length, batch_size, input_size).cuda()
output, (h_n, c_n) = model(input_data)

print(f"Output shape: {output.shape}")
print(f"h_n shape: {h_n.shape}")
print(f"c_n shape: {c_n.shape}")
