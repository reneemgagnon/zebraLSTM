import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import math
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

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

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_stripes, num_classes):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = StripedLSTM(embed_dim, hidden_dim, num_layers, num_stripes)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        return self.fc(hidden[-1])

# Data preparation
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

train_iter = IMDB(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

def collate_batch(batch):
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(vocab(tokenizer(_text)), dtype=torch.int64)
        text_list.append(processed_text)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = pad_sequence(text_list, batch_first=True, padding_value=0)
    return label_list.to(device), text_list.to(device)

# Hyperparameters
VOCAB_SIZE = len(vocab)
EMBED_DIM = 256
HIDDEN_DIM = 256
NUM_LAYERS = 2
NUM_STRIPES = 4
NUM_CLASSES = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 5

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model
model = SentimentClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_STRIPES, NUM_CLASSES).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Mixed precision training setup
scaler = GradScaler()

# Training loop
train_iter = IMDB(split='train')
test_iter = IMDB(split='test')

train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_iter, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_acc, total_count = 0, 0
    
    for idx, (labels, text) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        
        with autocast():
            predicted_label = model(text)
            loss = criterion(predicted_label, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_acc += (predicted_label.argmax(1) == labels).sum().item()
        total_count += labels.size(0)
        
        if idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {idx}, Loss: {loss.item():.4f}, Accuracy: {total_acc/total_count:.4f}')
    
    print(f'Epoch: {epoch}, Train Accuracy: {total_acc/total_count:.4f}')

# Evaluation
model.eval()
total_acc, total_count = 0, 0

with torch.no_grad():
    for idx, (labels, text) in enumerate(tqdm(test_dataloader)):
        predicted_label = model(text)
        total_acc += (predicted_label.argmax(1) == labels).sum().item()
        total_count += labels.size(0)

print(f'Test Accuracy: {total_acc/total_count:.4f}')