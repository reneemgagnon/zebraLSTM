"""
Striped LSTM Implementation
Copyright (c) 2024 Renee M Gagnon
This script is dual-licensed. See the LICENSE.md file for details.
For non-commercial and research use: CC BY-NC 4.0
For commercial use: Contact renee@freedomfamilyconsulting.ca
"""
#### 8. `tests/test_zebra_lstm.py`

```python
import torch
from src.zebra_lstm import StripedLSTM

def test_zebra_lstm():
    input_size = 100
    hidden_size = 200
    num_layers = 2
    num_stripes = 4
    batch_size = 32
    seq_length = 50

    model = StripedLSTM(input_size, hidden_size, num_layers, num_stripes).cuda()
    input_data = torch.randn(seq_length, batch_size, input_size).cuda()
    output, (h_n, c_n) = model(input_data)

    assert output.shape == (seq_length, batch_size, hidden_size)
    assert h_n.shape == (num_layers, batch_size, hidden_size)
    assert c_n.shape == (num_layers, batch_size, hidden_size)
