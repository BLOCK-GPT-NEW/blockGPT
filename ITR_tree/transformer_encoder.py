from torch import nn, Tensor
import time
import torch
import math
# Transformer 编码器模型1
class TransformerEncoderModel(nn.Module):
    def __init__(self,vocab_size,d_model, hidden_size, num_heads, num_layers):
        super(TransformerEncoderModel, self).__init__()
        self.encoder = nn.TransformerEncoder( nn.TransformerEncoderLayer(d_model, num_heads, hidden_size,batch_first=True) , num_layers)  
        self.output_layer = nn.Linear(d_model, vocab_size)
        #self.pos_encoder = PositionalEncoding(d_model, 0.1)
    def forward(self, x,src_mask,src_key_mask):
        x = self.encoder(x, mask = src_mask,src_key_padding_mask = src_key_mask)
        x = self.output_layer(x)
        return x

def generate_square_subsequent_mask(sz):
    #"""生成一个负无穷大的上三角矩阵，对角线上元素为 0。"""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Args:
#             x: Tensor, shape [seq_len, batch_size, embedding_dim]
#         """
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)