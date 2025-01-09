import torch
from torch import nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, dim, num_experts=4, topk=1):
        """
        Parameters:
            dim (int): The number of input feature channels.
            num_experts (int): The number of experts, default is 4.
            topk (int): The number of top-k experts to select, default is 1.
        """
        super().__init__()

        self.dwconv3 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim), LayerNorm2d(dim))
        self.dwconv5 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim), LayerNorm2d(dim))
        self.dwconv7 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim), LayerNorm2d(dim))
        self.dwconv9 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=9, stride=1, padding=4, groups=dim), LayerNorm2d(dim))

        self.jh = ConvolutionOp(in_channels=dim*4, out_channels=dim, groups = dim, hidden_c = dim*2)
        self.ln = LayerNorm2d(dim)   
        self.num_experts = num_experts
        self.topk = topk
        self.route = SampleRouter(c_in=dim, num_experts=self.num_experts)
        self.cond_pred = nn.ModuleList([nn.Conv2d(dim, 1, kernel_size=1) for _ in range(self.num_experts)])

    def forward(self, stage_features, training=False):
        """
        Parameters:
            stage_features (tuple): A tuple containing four feature maps from different stages, 
                                     i.e., (x_stage1, x_stage2, x_stage3, x_stage4).
            training (bool): If True, use Gumbel-Softmax for expert selection in training mode; 
                             otherwise, use top-k expert selection during inference.
        """         
        x_stage1, x_stage2, x_stage3, x_stage4 = stage_features
        B, C, H, W = x_stage1.shape

        x_stage1 = torch.chunk(x_stage1, 4, dim=1)
        x_stage2 = torch.chunk(x_stage2, 4, dim=1)
        x_stage3 = torch.chunk(x_stage3, 4, dim=1)
        x_stage4 = torch.chunk(x_stage4, 4, dim=1)
        
        x1 = self.dwconv3(torch.cat((x_stage1[0], x_stage2[0], x_stage3[0], x_stage4[0]), dim=1))
        x2 = self.dwconv5(torch.cat((x_stage1[1], x_stage2[1], x_stage3[1], x_stage4[1]), dim=1))
        x3 = self.dwconv7(torch.cat((x_stage1[2], x_stage2[2], x_stage3[2], x_stage4[2]), dim=1))
        x4 = self.dwconv9(torch.cat((x_stage1[3], x_stage2[3], x_stage3[3], x_stage4[3]), dim=1))

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.jh(x)
        x = self.ln(x)

        weights = self.route(x)     
        if training:
            gate = F.gumbel_softmax(weights, hard=True, dim=-1)
        else:
            _, topk_indices = torch.topk(weights, self.topk, dim=1)
            gate = torch.zeros_like(weights).scatter_(1, topk_indices, 1)
        
        expert_outputs = torch.stack([cond(x) for cond in self.cond_pred], dim=1)
        weighted_expert_outputs = gate.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * expert_outputs
        res = weighted_expert_outputs.sum(dim=1) 
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
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
    
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
class ConvolutionOp(nn.Module):
  def __init__(self, 
               in_channels : int = 144, 
               out_channels : int = 12, 
               groups : int = 12, 
               hidden_c : int = 96
               ): 
    super().__init__()

    self.conv_inner_subspace_1 = nn.Conv2d(in_channels = in_channels, out_channels = hidden_c, kernel_size = 1, groups = groups)
    self.conv_inner_subspace_2 = nn.Conv2d(in_channels = hidden_c, out_channels = out_channels, kernel_size = 1, groups = groups)
    
  def forward(self, x):
    x = self.conv_inner_subspace_2(nn.ReLU()(self.conv_inner_subspace_1(x)))
    return x