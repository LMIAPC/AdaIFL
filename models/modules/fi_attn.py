import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class FIA(nn.Module):
    def __init__(self, dim, num_heads=8, total_token_nums=320, num_experts=4, topk=2, qkv_bias=False, qk_scale=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.total_token_nums = total_token_nums
        self.num_experts = num_experts
        self.topk = topk
       
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_linears = nn.ModuleList([nn.Linear(dim, dim * 2, bias=qkv_bias) for _ in range(self.num_experts)])
        self.route = SampleRouter(c_in= dim, num_experts=self.num_experts)
        self.proj = nn.Linear(dim, dim)


    def cluster_tokens(self, x, cluster_num, k):
        """
        Cluster tokens using the Density Peak Clustering with K-Nearest Neighbors (DPC-KNN) algorithm.
        For more details, see: https://www.sciencedirect.com/science/article/abs/pii/S0950705116000794?via%3Dihub.
        """
        N, C = x.shape
        dist_matrix = torch.cdist(x.unsqueeze(0), x.unsqueeze(0)) / (C ** 0.5)
        dist_matrix = dist_matrix.squeeze(0)
        dist_nearest, _ = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        density += torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6

        mask = density[None, :] > density[:, None]
        mask = mask.type(x.dtype)
        dist_max = dist_matrix.flatten().max()[None, None]
        dist, _ = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        score = dist * density
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)
        dist_matrix = dist_matrix[index_down]
        idx_cluster = dist_matrix.argmin(dim=0)
        idx_cluster[index_down] = torch.arange(cluster_num, device=x.device)

        return idx_cluster
    
    def merge_tokens(self, x, idx_cluster, cluster_num, token_weight=None):
        """
        Merge tokens based on DPC-KNN cluster assignments and weighted aggregation.
        """
        N, C = x.shape
        if token_weight is None:
            token_weight = x.new_ones(N, 1)

        all_weight = token_weight.new_zeros(cluster_num, 1)
        all_weight.index_add_(dim=0, index=idx_cluster, source=token_weight)
        all_weight = all_weight + 1e-6
        norm_weight = token_weight / all_weight[idx_cluster]

        x_merged = x.new_zeros(cluster_num, C)
        source = x * norm_weight
        x_merged.index_add_(dim=0, index=idx_cluster, source=source.type(x.dtype))
        return x_merged

    def aggregation_tokens(self, x, num_clusters, token_weight, k): 
        """ 
        Args:
            x : Token features (N tokens, C features).
            num_clusters : Number of clusters to form.
            token_weight : Weight for each token.
            k : Number of nearest neighbors for clustering (used by DPC-KNN).
        Returns:
            Tensor[num_clusters, C]: Aggregated features for each cluster.
        """     
        buckets = self.cluster_tokens(x, num_clusters, k)
        x_agg = self.merge_tokens(x, buckets, num_clusters, token_weight)
        return x_agg
    
    def region_partition_and_scale_allocation(self, x, score_pred, R1_scale_pred, R2_scale_pred, R3_scale_pred):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        score = score_pred(x).exp().squeeze(-1)
        score_sort, score_sort_index = torch.sort(score, dim=1)

        def get_partitioned_tokens(start_idx, end_idx):
            return torch.gather(x, 1, score_sort_index[:, start_idx:end_idx].unsqueeze(-1).repeat(1, 1, C))
        
        R1_tokens = get_partitioned_tokens(0, H * W // 3)
        R2_tokens = get_partitioned_tokens(H * W // 3, H * W // 3 * 2)
        R3_tokens = get_partitioned_tokens(H * W // 3 * 2, H * W)

        def get_scoremap_and_scale(scoremap, scale_pred_func):
            scale = scale_pred_func(scoremap)
            scoremap = scoremap.unsqueeze(2)
            return scoremap, scale
        
        R1_scoremap, R1_scale = get_scoremap_and_scale(score_sort[:, :H * W // 3], R1_scale_pred)
        R2_scoremap, R2_scale = get_scoremap_and_scale(score_sort[:, H * W // 3:H * W // 3 * 2], R2_scale_pred)
        R3_scoremap, R3_scale = get_scoremap_and_scale(score_sort[:, H * W // 3 * 2:], R3_scale_pred)

        total_scale = torch.cat([R1_scale, R2_scale, R3_scale], dim=1)
        normalized_scale = F.softmax(total_scale, dim=1)
        R1_scale, R2_scale, R3_scale = torch.split(normalized_scale, 1, dim=1)
        R1_scale = R1_scale.squeeze(dim=1)
        R2_scale = R2_scale.squeeze(dim=1)
        R3_scale = R3_scale.squeeze(dim=1)

        def calculate_agg_num(scale):
            return torch.clamp(self.total_token_nums * scale, min=16, max=self.total_token_nums)

        R1_agg_num = calculate_agg_num(R1_scale)
        R2_agg_num = calculate_agg_num(R2_scale)
        R3_agg_num = calculate_agg_num(R3_scale)

        return R1_tokens, R2_tokens, R3_tokens, R1_agg_num, R2_agg_num, R3_agg_num, R1_scoremap, R2_scoremap, R3_scoremap
    
    def forward(self, x, score_pred, R1_scale_pred, R2_scale_pred, R3_scale_pred):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        R1_tokens, R2_tokens, R3_tokens, R1_agg_num, R2_agg_num, R3_agg_num, R1_scoremap, R2_scoremap, R3_scoremap = self.region_partition_and_scale_allocation(x, score_pred, R1_scale_pred, R2_scale_pred, R3_scale_pred)


        weights = self.route(x)
        _, topk_indices = torch.topk(weights, self.topk, dim=1)

        results = []
        for i in range(B):
            R1_num = int(R1_agg_num[i].item())
            R2_num = int(R2_agg_num[i].item())
            R3_num = int(R3_agg_num[i].item())    
            agg_tokens_R1 = self.aggregation_tokens(R1_tokens[i], R1_num, R1_scoremap[i], k=int(math.sqrt(R1_num)))  #b,n,c
            agg_tokens_R2 = self.aggregation_tokens(R2_tokens[i], R2_num, R2_scoremap[i], k=int(math.sqrt(R2_num)))  #b,n,c
            agg_tokens_R3 = self.aggregation_tokens(R3_tokens[i], R3_num, R3_scoremap[i], k=int(math.sqrt(R3_num)))  #b,n,c
            agg_tokens = torch.cat([agg_tokens_R1, agg_tokens_R2, agg_tokens_R3], dim=0)
            
            outputs = []
            for idx in topk_indices[i]:
                kv = self.kv_linears[idx](agg_tokens)
                kv = kv.reshape(-1, 2, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
                k, v = kv[0], kv[1]
                attn = (q[i] @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                output = (attn @ v).transpose(1, 2)
                
                output = output.reshape(C, N).permute(1, 0)
                outputs.append(output * weights[i, idx])
            results.append(sum(outputs))
        res = torch.stack(results)
        res = self.proj(res)
        return res

    
class SampleRouter(nn.Module):
    """
    Parameters:
        c_in (int): The number of input channels (features).
        num_experts (int): The number of experts the input features can be routed to.
    """
    def __init__(self, c_in, num_experts):
        super(SampleRouter, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(c_in, num_experts)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
                
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x