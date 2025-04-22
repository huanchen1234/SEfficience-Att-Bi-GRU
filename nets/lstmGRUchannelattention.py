import torch
import torch.nn as nn

def bidirectional_rnn(input_data, forward_rnn_type='GRU', backward_rnn_type='LSTM', hidden_size=20, reduction_ratio=16):
    device = input_data.device  # 获取输入数据的设备信息
    # 创建正向和逆向的循环层
    if forward_rnn_type == 'GRU':
        forward_rnn = nn.GRU(1000, hidden_size, batch_first=True).to(device)
    elif forward_rnn_type == 'LSTM':
        forward_rnn = nn.LSTM(1000, hidden_size, batch_first=True).to(device)
    else:
        raise ValueError("Invalid forward RNN type")

    if backward_rnn_type == 'GRU':
        backward_rnn = nn.GRU(1000, hidden_size, batch_first=True).to(device)
    elif backward_rnn_type == 'LSTM':
        backward_rnn = nn.LSTM(1000, hidden_size, batch_first=True).to(device)
    else:
        raise ValueError("Invalid backward RNN type")
    
    # 将输入数据移动到相同的设备上
    input_data = input_data.to(device)

    # 正向计算
    forward_output, _ = forward_rnn(input_data)

    # 逆向计算
    reverse_input_data = torch.flip(input_data, dims=[1])  # 反转输入序列
    backward_output, _ = backward_rnn(reverse_input_data)

    # 合并正向和逆向的输出
    bidirectional_output = torch.cat([forward_output, torch.flip(backward_output, dims=[1])], dim=1)
    print(len(bidirectional_output.size()))
    # 通道注意力机制
    # b, seq_len, _ = bidirectional_output.size()  # 获取维度大小
    b = bidirectional_output.size(0)
    seq_len = bidirectional_output.size(1)
    avg_pool = nn.AdaptiveAvgPool1d(seq_len)
    c = bidirectional_output.size(2)  # 获取通道维度大小    
    fc1 = nn.Linear(c, c // reduction_ratio).to(device)
    relu = nn.ReLU()
    fc2 = nn.Linear(c // reduction_ratio, c).to(device)
    sigmoid = nn.Sigmoid()
    bidirectional_output = bidirectional_output.unsqueeze(3)  # 在第三维度上添加一个维度
    attention_weights = sigmoid(fc2(relu(fc1(avg_pool(bidirectional_output))).squeeze(2)))

    attention_weights = attention_weights.view(b, seq_len, 1, 1)

    output_with_attention = bidirectional_output * attention_weights

    return output_with_attention




