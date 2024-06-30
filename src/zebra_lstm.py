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
        for i in range(self.num_stripes):
            nn.init.kaiming_uniform_(self.weight_ih[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.weight_hh[i], a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_ih[i])
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_ih[i], -bound, bound)
            nn.init.uniform_(self.bias_hh[i], -bound, bound)

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1 = []
        c_1 = []

        for i in range(self.num_stripes):
            h_0_stripe = h_0[:, i*self.stripe_size:(i+1)*self.stripe_size]
            c_0_stripe = c_0[:, i*self.stripe_size:(i+1)*self.stripe_size]

            gates = F.linear(input, self.weight_ih[i], self.bias_ih[i]) + \
                    F.linear(h_0_stripe, self.weight_hh[i], self.bias_hh[i])

            i_gate, f_gate, g_gate, o_gate = gates.chunk(4, 1)

            i_gate = torch.sigmoid(i_gate)
            f_gate = torch.sigmoid(f_gate)
            g_gate = torch.tanh(g_gate)
            o_gate = torch.sigmoid(o_gate)

            c_1_stripe = f_gate * c_0_stripe + i_gate * g_gate
            h_1_stripe = o_gate * torch.tanh(c_1_stripe)

            h_1.append(h_1_stripe)
            c_1.append(c_1_stripe)

        h_1 = torch.cat(h_1, dim=1)
        c_1 = torch.cat(c_1, dim=1)

        return h_1, c_1
