import torch
import torch.nn as nn

def bidirectional_rnn_with_attention(input_data, forward_rnn_type='GRU', backward_rnn_type='LSTM', hidden_size=20):
    device = input_data.device  # 获取输入数据的设备信息

    # 创建正向和逆向的循环层，并将其移动到相同的设备上
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

    # 计算注意力权重
    attention_scores = torch.matmul(forward_output, backward_output.transpose(2, 1))
    attention_weights = torch.softmax(attention_scores, dim=2)

    # 根据注意力权重计算上下文向量
    context_vector = torch.matmul(attention_weights.transpose(1, 2), forward_output)

    # 合并正向和逆向的输出
    bidirectional_output = torch.cat([forward_output, torch.flip(backward_output, dims=[1])], dim=2)

    # 应用注意力机制后的输出
    output = torch.cat([bidirectional_output, context_vector], dim=2)

    return output

