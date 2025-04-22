import torch
import torch.nn as nn

def bidirectional_rnn_with_attention(input_data, forward_rnn_type='GRU', backward_rnn_type='LSTM', hidden_size=20):
    input_size = input_data.size(-1)
    seq_length = input_data.size(1)

    if forward_rnn_type == 'GRU':
        forward_rnn = nn.GRU(input_size, hidden_size, batch_first=True).to(input_data.device)
    elif forward_rnn_type == 'LSTM':
        forward_rnn = nn.LSTM(input_size, hidden_size, batch_first=True).to(input_data.device)
    else:
        raise ValueError("Invalid forward RNN type")

    if backward_rnn_type == 'GRU':
        backward_rnn = nn.GRU(input_size, hidden_size, batch_first=True).to(input_data.device)
    elif backward_rnn_type == 'LSTM':
        backward_rnn = nn.LSTM(input_size, hidden_size, batch_first=True).to(input_data.device)
    else:
        raise ValueError("Invalid backward RNN type")

    forward_output, _ = forward_rnn(input_data)
    reverse_input_data = torch.flip(input_data, dims=[1])
    backward_output, _ = backward_rnn(reverse_input_data)

    bidirectional_output = torch.cat([forward_output, torch.flip(backward_output, dims=[1])], dim=1)

    
    attn_weights = torch.softmax(bidirectional_output)



    context_vector = torch.matmul(attn_weights, bidirectional_output)

    return context_vector








