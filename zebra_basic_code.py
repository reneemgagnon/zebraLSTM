import torch

# Initialize the model
model = StripedLSTM(input_size, hidden_size, num_layers, num_stripes).cuda()

# Move input data to GPU
input_data = torch.randn(seq_length, batch_size, input_size).cuda()

# Initialize hidden states on GPU
hidden = (torch.zeros(num_layers, batch_size, hidden_size).cuda(),
          torch.zeros(num_layers, batch_size, hidden_size).cuda())

# Run the model on GPU
output, (h_n, c_n) = model(input_data, hidden)

print(f"Output shape: {output.shape}")
print(f"h_n shape: {h_n.shape}")
print(f"c_n shape: {c_n.shape}")
