import torch
from torch import nn, Tensor
# from labml_helpers.module import Module
# from labml_nn.transformers.feed_forward import FeedForward
# from labml_nn.transformers.mha import MultiHeadAttention
# from labml_nn.utils import clone_module_list
import torch.nn.functional as F
# from group import Group
# from pointNet import Pointnet
from einops import rearrange, repeat
# from timm.models.layers import DropPath

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


class CrossAttentionModule(torch.nn.Module):
    def __init__(self, embed_size, num_heads):
        super(CrossAttentionModule, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads

        # 使用线性层来生成 Q, K, V
        self.query_linear = torch.nn.Linear(embed_size, embed_size)  # 生成 Q
        self.key_linear = torch.nn.Linear(embed_size, embed_size)  # 生成 K
        self.value_linear = torch.nn.Linear(embed_size, embed_size)  # 生成 V

        self.attn_dropout = torch.nn.Dropout(0.1)
        self.output_linear = torch.nn.Linear(embed_size, embed_size)

    def forward(self, pts, cubesize):
        # pts 和 cubesize 都是 [B, N, 128] 的张量
        batch_size = pts.size(0)

        # 线性变换生成 Q, K, V
        Q = self.query_linear(cubesize)  # [B, 128, C] 以 cubesize 作为 Q
        K = self.key_linear(pts)  # [B, 128, C] 以 pts 作为 K
        V = self.value_linear(pts)  # [B, 128, C] 以 pts 作为 V

        # 计算注意力得分 (点积注意力)
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [B, 128, 128]（点积结果）

        # 缩放
        scores = scores / (self.embed_size ** 0.5)

        # 应用 Softmax 来获得注意力权重
        attention_weights = F.softmax(scores, dim=-1)  # [B, 128, 128]
        attention_weights = self.attn_dropout(attention_weights)

        # 使用注意力权重加权求和
        output = torch.matmul(attention_weights, V)  # [B, 128, C]

        # 通过线性层输出
        output = self.output_linear(output)  # [B, 128, C]

        return output



class Physics_Attention_Irregular_Mesh(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape

        ### (1) Slice
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)######[B, heads, slice_num, dim_head]
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')#####[B, N, inner_dim]
        return self.to_out(out_x)#####[B, N, 128]


class PointNet(nn.Module):
    # Embedding module
    def __init__(self, inchannel, encoder_channel):
        super(PointNet, self).__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(inchannel, 128, 1),
            nn.BatchNorm1d(128),
            # nn.GroupNorm(num_groups=1, num_channels=128),  # 使用GroupNorm
            nn.GELU(),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            # nn.GroupNorm(num_groups=1, num_channels=512),  # 使用GroupNorm
            nn.GELU(),
            nn.Conv1d(512, encoder_channel, 1)
        )

    def forward(self, point_groups):
        bs, c, n = point_groups.shape  # B 6 N
        # encoder
        feature = self.first_conv(point_groups)  # BG 256 N

        # 获取最大特征
        feature_global, _ = torch.max(feature, dim=2, keepdim=True)  # BG 256 1
        feature_global_expanded = feature_global.expand(-1, -1, n)  # BG 256 N
        feature = torch.cat([feature_global_expanded, feature], dim=1)  # BG 512 N

        feature = self.second_conv(feature)  # BG 512 N
        return feature  # (B, G, C, N)


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_block(nn.Module):
    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=16,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Irregular_Mesh(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                     dropout=dropout, slice_num=slice_num)
        self.LN2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)
        self.ln_4 = nn.Linear(hidden_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.bn_1 = nn.BatchNorm1d(10000)

    def forward(self, fx):
        # fx = self.Attn(self.ln_1(fx)) + fx
        # fx = self.mlp(self.ln_2(fx)) + fx
        fa = self.Attn(self.ln_1(fx))#####[B, N, 128]
        fa = self.gelu(self.LN2(self.ln_4(fx - fa))) + fx  ### offset LBA
        fx = self.mlp(fa) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx



class phsoffNet(nn.Module):
    def __init__(
            self,
            dim: int,
            dropout: float = 0.2,  # 0.2
            n_layers=4,
            n_hidden=128,
            # dropout=0,
            n_head=8,
            act='gelu',
            mlp_ratio=1,
            out_dim=1,
            slice_num=16,

    ):
        super().__init__()

        self.flow_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Linear(512, 128),
            # nn.GELU(),
            # nn.Linear(256, 256),
        )

        self.point_net = PointNet(inchannel=3, encoder_channel=128)
        self.cross_attention = CrossAttentionModule(embed_size=128, num_heads=8)
        self.blocks = nn.ModuleList([Transolver_block(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio,
                                                      out_dim=out_dim,
                                                      slice_num=slice_num,
                                                      last_layer=(_ == n_layers - 1))

                                     for _ in range(n_layers)])

    def forward(self, pts, cubesize):
        B, N, C = pts.shape
        pts = pts.permute(0, 2, 1)  # [B, C ,N]
        x = self.point_net(pts)  ### [B,128,N]
        x = x.permute(0, 2, 1)  # [B, N ,128]
        cubesize = self.flow_net(cubesize)  # [B, N ,128] ##
        fx = self.cross_attention(x, cubesize) ##### [B,N,128]

        for block in self.blocks:
            fx = block(fx)  #####[B,N,1]

        x_pre = fx.squeeze(dim=-1)  # [B, N]
        x_pre = x_pre.mean(dim=1, keepdim=True)
        x_pre = x_pre.squeeze(dim=-1)  # [B, N]

        return x_pre
