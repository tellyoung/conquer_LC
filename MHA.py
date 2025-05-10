class MHA:
    def __init__(self):
        self.num_heads = num_heads
        self.d_k = dim // num_heads

        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.W_o = nn.Linear(dim, dim)

    def forward(self, Q, K, V):
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        Q = split_heads(self.num_heads, self.d_k)
        K = split_heads(self.num_heads, self.d_k)
        V = split_heads(self.num_heads, self.d_k)

        Q, K, V = pse(Q, K, V)

        attn_scores = matmul(Q, K.T) / (self.d_k ** 0.5)

        attn_scores = mask(attn_scores)

        attn_scores = softmax(attn_scores)
        
        output = matmul(attn_scores, V)

        output = concat(output)

        output = self.W_o(output)

        return output


#########################################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """正弦位置编码（原始Transformer论文方案）"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        # 生成位置编码矩阵 (max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)          # 位置索引列向量
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置用cos
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        """
        Args:
            x: 输入张量 (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]  # 自动匹配当前序列长度
        return self.dropout(x)

class LearnablePositionalEmbedding(nn.Module):
    """可学习的位置编码（更适用于固定长度场景）"""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.embed = nn.Embedding(max_len, d_model)
        self.register_buffer('position_ids', torch.arange(max_len))
        
    def forward(self, x):
        pos = self.position_ids[:x.size(1)]  # 获取当前序列的位置id
        return x + self.embed(pos)  # (batch, seq, d_model)



class MultiHeadAttention(nn.Module):
    def __init__(self, dim=512, h=8, dropout=0.1):
        """
        Args:
            dim: 模型维度（默认512）
            h: 注意力头数（默认8）
            dropout: dropout概率
        """
        super().__init__()
        assert dim % h == 0, "dim必须能被h整除"
        
        self.d_k = dim // h  # 每个头的维度
        self.h = h
        
        # 使用单个线性层并行计算所有头的Q/K/V
        self.qkv_proj = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: 输入张量 (batch_size, seq_len, dim)
            mask: 掩码张量 (batch_size, 1, 1, seq_len) 或 (batch_size, seq_len)
        Returns:
            输出张量 (batch_size, seq_len, dim)
        """
        x = PositionalEncoding(x)

        batch_size, seq_len, _ = x.size()
        
        # 并行计算所有Q/K/V → (batch, seq_len, 3 * dim)
        qkv = self.qkv_proj(x)
        
        # 拆分多头：先展开最后维度到 (h, d_k) → 转置维度方便计算
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)  # (b, h, seq, d_k)
        k = k.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        
        # 计算缩放点积注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)  # (b, h, seq, seq)
        
        # 应用mask（可选）
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # 扩展头维度
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和 → (b, h, seq, d_k)
        context = torch.matmul(attn_weights, v)
        
        # 合并多头 → (b, seq, dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # 最终投影
        return self.out_proj(context)

# ---------- 使用示例 ----------
if __name__ == "__main__":
    mha = MultiHeadAttention(dim=512, h=8)
    x = torch.randn(2, 10, 512)  # (batch_size, seq_len, dim)
    output = mha(x)
    print(output.shape)  # torch.Size([2, 10, 512])