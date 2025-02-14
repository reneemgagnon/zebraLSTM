"""
Striped LSTM Implementation
Copyright (c) 2024 Renee M Gagnon
This script is dual-licensed. See the LICENSE.md file for details.
For non-commercial and research use: CC BY-NC 4.0
For commercial use: Contact renee@freedomfamilyconsulting.ca
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class StripedLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_stripes):
        super(StripedLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_stripes = num_stripes
        self.stripe_size = hidden_size // num_stripes

        # Create weight matrices for each stripe
        self.weight_ih = nn.ParameterList([
            nn.Parameter(torch.Tensor(4 * self.stripe_size, input_size))
            for _ in range(num_stripes)
        ])
        self.weight_hh = nn.ParameterList([
            nn.Parameter(torch.Tensor(4 * self.stripe_size, self.stripe_size))
            for _ in range(num_stripes)
        ])
        self.bias_ih = nn.ParameterList([
            nn.Parameter(torch.Tensor(4 * self.stripe_size))
            for _ in range(num_stripes)
        ])
        self.bias_hh = nn.ParameterList([
            nn.Parameter(torch.Tensor(4 * self.stripe_size))
            for _ in range(num_stripes)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using the method described in "Delving deep into rectifiers" """
        for i in range(self.num_stripes):
            nn.init.kaiming_uniform_(self.weight_ih[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.weight_hh[i], a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_ih[i])
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_ih[i], -bound, bound)
            nn.init.uniform_(self.bias_hh[i], -bound, bound)

    def forward(self, input, hidden):
        """
        Forward pass of the StripedLSTMCell.
        
        Args:
            input (Tensor): Input tensor of shape (batch_size, input_size)
            hidden (tuple): Tuple containing (h_0, c_0) of shape (batch_size, hidden_size)
        
        Returns:
            tuple: New (h_1, c_1) tensors
        """
        h_0, c_0 = hidden
        h_1 = []
        c_1 = []

        for i in range(self.num_stripes):
            # Get the stripe of the hidden state and cell state
            h_0_stripe = h_0[:, i*self.stripe_size:(i+1)*self.stripe_size]
            c_0_stripe = c_0[:, i*self.stripe_size:(i+1)*self.stripe_size]

            # Compute gate inputs
            gates = F.linear(input, self.weight_ih[i], self.bias_ih[i]) + \
                    F.linear(h_0_stripe, self.weight_hh[i], self.bias_hh[i])

            # Slice gates into i, f, g, o
            i_gate, f_gate, g_gate, o_gate = gates.chunk(4, 1)

            # Apply activations
            i_gate = torch.sigmoid(i_gate)
            f_gate = torch.sigmoid(f_gate)
            g_gate = torch.tanh(g_gate)
            o_gate = torch.sigmoid(o_gate)

            # Compute new cell state and hidden state
            c_1_stripe = f_gate * c_0_stripe + i_gate * g_gate
            h_1_stripe = o_gate * torch.tanh(c_1_stripe)

            h_1.append(h_1_stripe)
            c_1.append(c_1_stripe)

        # Concatenate stripes
        h_1 = torch.cat(h_1, dim=1)
        c_1 = torch.cat(c_1, dim=1)

        return h_1, c_1

class StripedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_stripes, dropout=0):
        super(StripedLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_stripes = num_stripes
        self.dropout = dropout

        # Create LSTM layers
        self.lstm_layers = nn.ModuleList([
            StripedLSTMCell(
                input_size if i == 0 else hidden_size,
                hidden_size,
                num_stripes
            )
            for i in range(num_layers)
        ])

    def forward(self, input, hidden=None):
        """
        Forward pass of the StripedLSTM.
        
        Args:
            input (Tensor): Input tensor of shape (seq_len, batch_size, input_size)
            hidden (tuple): Initial hidden state. If None, initialized with zeros.
        
        Returns:
            tuple: (output, (h_n, c_n)) where output is the output of the last layer for each time step,
                   and (h_n, c_n) is the final hidden state for each layer
        """
        is_packed = isinstance(input, nn.utils.rnn.PackedSequence)
        if is_packed:
            input, batch_sizes = input.data, input.batch_sizes
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = input.size(1)

        if hidden is None:
            hidden = self.init_hidden(max_batch_size)

        h_n = []
        c_n = []
        output = []

        for layer in range(self.num_layers):
            layer_output = []
            h_i, c_i = hidden[layer]

            for t in range(input.size(0)):
                if is_packed:
                    t_input = input[t, :batch_sizes[t]]
                    t_h_i = h_i[:batch_sizes[t]]
                    t_c_i = c_i[:batch_sizes[t]]
                else:
                    t_input = input[t]
                    t_h_i = h_i
                    t_c_i = c_i

                h_i, c_i = self.lstm_layers[layer](t_input, (t_h_i, t_c_i))
                layer_output.append(h_i)

                if self.dropout > 0 and layer < self.num_layers - 1:
                    h_i = F.dropout(h_i, p=self.dropout, training=self.training)

            layer_output = torch.stack(layer_output)
            input = layer_output
            h_n.append(h_i)
            c_n.append(c_i)

        h_n = torch.stack(h_n)
        c_n = torch.stack(c_n)

        if is_packed:
            output = nn.utils.rnn.PackedSequence(input, batch_sizes)
        else:
            output = input

        return output, (h_n, c_n)

    def init_hidden(self, batch_size):
        """Initialize hidden states"""
        weight = next(self.parameters()).data
        return [(weight.new(batch_size, self.hidden_size).zero_(),
                 weight.new(batch_size, self.hidden_size).zero_())
                for _ in range(self.num_layers)]

# Usage example
input_size = 100
hidden_size = 200
num_layers = 2
num_stripes = 4
batch_size = 32
seq_length = 50

model = StripedLSTM(input_size, hidden_size, num_layers, num_stripes)
input_data = torch.randn(seq_length, batch_size, input_size)
output, (h_n, c_n) = model(input_data)

print(f"Output shape: {output.shape}")
print(f"h_n shape: {h_n.shape}")
print(f"c_n shape: {c_n.shape}")