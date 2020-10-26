import torch
import torch.nn as nn


# 测试
class TextRNN(nn.Module):
    def __init__(self, length):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(length, 100)
        self.lstm = nn.GRU(100, 128, batch_first=True, bidirectional=True)  #
        self.linear = nn.Linear(256, 2)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # print(x.size())
        # print(x)
        vec = self.embedding(x)
        # print(vec.size())
        # print(vec)
        res, h_n = self.lstm(vec)
        # print(res.size())
        # print(res)
        print(h_n.size())
        print(h_n)
        # final_out = self.linear(res[:, -1, :])
        # final_out = F.softmax(final_out)
        feature_map = torch.cat([h_n[i, :, :] for i in range(h_n.shape[0])],
                                dim=1)  # 64*256 Batch_size * (hidden_size * hidden_layers * 2)
        # print(feature_map.size())
        final_out = self.linear(feature_map)
        # print(final_out.size())
        final_out = self.softmax(final_out)
        # print(final_out.size())
        return final_out
