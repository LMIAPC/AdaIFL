import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Type
from .fi_attn import FIA
from .moe import MoE
from .dynamic_decoder import Decoder

"""
Thanks for the open-source of SAM, part of codes are from their implementation:
https://github.com/facebookresearch/segment-anything
"""

class adaifl_model(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Number of the transformer-based dynamic encoder layers.
            num_heads (int): Number of attention heads.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.global_attn_indexes = global_attn_indexes
        # Patch embedding layer.
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        # Scoring network of region features.
        self.score_pred = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//4),           
            nn.GELU(),
            nn.Linear(embed_dim//4, 1),
            )
        
        # Aggregation scale allocation. In the implementation of our model, three sub-regions are defined.
        self.R1_scale_pred = EqualLinear(1365, 1, bias_init_val=1)
        self.R2_scale_pred = EqualLinear(1365, 1, bias_init_val=1)
        self.R3_scale_pred = EqualLinear(1366, 1, bias_init_val=1)

        # Dynamic encoder.
        self.encoder = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                cur_depth = i,
                input_size=(img_size // patch_size, img_size // patch_size),
                global_attn_indexes=global_attn_indexes
            )
            self.encoder.append(block)
        
        # Dynamic decoder.
        self.decoder = Decoder(dim=embed_dim)
        
    def forward(self, x):
        stage_features = []
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
    
        for layer_idx, blk in enumerate(self.encoder):
            x = blk(x, self.score_pred, self.R1_scale_pred, self.R2_scale_pred, self.R3_scale_pred)
            if layer_idx in self.global_attn_indexes:
                stage_features.append(x.permute(0, 3, 1, 2))

        mask_pred = self.decoder(stage_features)
        return mask_pred


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        cur_depth: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            cur_depth: Current block depth.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.cur_depth = cur_depth
        self.global_attn_indexes = global_attn_indexes
        self.norm1 = norm_layer(dim)
       
        if cur_depth in self.global_attn_indexes:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                input_size=input_size,
            )
        else:
            self.fi_attn = FIA(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=None)

        self.norm2 = norm_layer(dim)

        self.feature_moe = MoE(
            input_size=dim, 
            head_size=dim//2,
            output_size=dim,
            num_experts=6, 
            top_k=2, 
            bias=False, 
            activation=NewGELU,
            acc_aux_loss=False,
            gating_dropout=0.0,
            sample_topk=0,
            gating_size=256,
            aux_loss='mi',
            gate_type="mlp",
            )
        
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
    
    def forward(self, x, score_pred, R1_scale_pred, R2_scale_pred, R3_scale_pred):
        b_, h_, w_, c_ = x.shape

        shortcut = x
        x = self.norm1(x)
        if self.cur_depth in self.global_attn_indexes:
            x = self.attn(x)   
        else:
            x = x.reshape(b_, h_ * w_, c_)
            x = self.fi_attn(x, score_pred, R1_scale_pred, R2_scale_pred, R3_scale_pred)
            x = x.reshape(b_, h_, w_, c_)
             
        x = shortcut + x
        xn = self.norm2(x)
        x_mlp = self.mlp(xn)
        x_moe = xn.reshape(xn.size(0), -1, xn.size(3))
        x_moe, _, _ = self.feature_moe(x_moe)
        x_moe = x_moe.view(b_, h_, w_, c_)
        x = x + x_moe + x_mlp
        return x

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))
    
class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))


    def forward(self, x) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)  

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x

class EqualLinear(nn.Module):
    def __init__(self, in_channels, out_channels, bias_init_val=0, lr_mul=1):
        super(EqualLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lr_mul = lr_mul
        
        self.scale = (1 / math.sqrt(in_channels)) * lr_mul
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(out_channels).fill_(bias_init_val))

    def forward(self, x):
        bias = self.bias * self.lr_mul
        out = F.linear(x, self.weight * self.scale, bias=bias)
        return out

@torch.jit.script
def NewGELU(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x