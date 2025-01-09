import torch
import torch.nn as nn
from functools import partial
from .modules import adaifl_model


class AdaIFL(nn.Module):
    def __init__(self,
                 encoder_embed_dim = 768,
                 encoder_depth = 12,
                 encoder_num_heads = 12,
                 encoder_global_attn_indexes = [2, 5, 8, 11],
                 image_size = 1024,
                 vit_patch_size = 16,
                 ):   
        """
        Args:
            encoder_embed_dim (int): Embedding dimension of the transformer-based dynamic encoder.
            encoder_depth (int): Number of layers in the encoder.
            encoder_num_heads (int): Number of attention heads in each encoder layer.
            encoder_global_attn_indexes (list of int): List of layer indices that use global attention.
            image_size (int): Input image size.
            vit_patch_size (int): The size of each patch in the transformer-based dynamic encoder.
        """
        super().__init__()
        
        self.model = adaifl_model(
            depth = encoder_depth,
            embed_dim = encoder_embed_dim,
            img_size = image_size,
            mlp_ratio = 4,
            norm_layer = partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads = encoder_num_heads,
            patch_size = vit_patch_size,
            qkv_bias = True,
            use_rel_pos = True,
            global_attn_indexes = encoder_global_attn_indexes,
            )
 
    def forward(self, x):
        out = self.model(x)
        return out