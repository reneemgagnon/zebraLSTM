import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

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

class StripedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_stripes, dropout=0):
        super(StripedLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_stripes = num_stripes
        self.dropout = dropout

        self.lstm_layers = nn.ModuleList([
            StripedLSTMCell(
                input_size if i == 0 else hidden_size,
                hidden_size,
                num_stripes
            )
            for i in range(num_layers)
        ])

    def forward(self, input, hidden=None):
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
        weight = next(self.parameters()).data
        return [(weight.new(batch_size, self.hidden_size).zero_(),
                 weight.new(batch_size, self.hidden_size).zero_())
                for _ in range(self.num_layers)]

# Configuration and setup
input_size = 100
hidden_size = 200
num_layers = 2
num_stripes = 4
batch_size = 32
seq_length = 50
learning_rate = 0.001
num_epochs = 10

# Initialize the model
model = StripedLSTM(input_size, hidden_size, num_layers, num_stripes).cuda()

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Mixed precision training setup
scaler = GradScaler()

# Dummy data loader
input_data = torch.randn(seq_length, batch_size, input_size).cuda()
target_data = torch.randint(0, input_size, (seq_length, batch_size)).cuda()

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    with autocast():
        output, _ = model(input_data)
        loss = criterion(output.view(-1, output.size(-1)), target_data.view(-1))

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

print("Training completed successfully.")
