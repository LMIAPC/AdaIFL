import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd
from typing import Any, Dict, List, Optional
from torch import Tensor

"""
Implementation borrowed from ModuleFormer.
https://github.com/IBM/ModuleFormer/tree/main/moduleformer/utils
Thanks for the open-source environment.
"""

class MoE(nn.Module):
    """
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """
    def __init__(
        self, 
        input_size, 
        head_size,
        output_size,
        num_experts, 
        top_k,
        bias=False, 
        activation=None, 
        acc_aux_loss=False,
        hidden_size=None,
        gating_dropout=0.0,
        sample_topk=0,
        gating_size=256,
        aux_loss='mi',
        gate_type='mlp',
        ):
        super(MoE, self).__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        self.head_size = head_size
        self.output_size = output_size
        self.bias = bias
        self.experts = ParallelExperts(num_experts, input_size, head_size, bias)
        if hidden_size is None:
            hidden_size = head_size
        self.output_experts = ParallelExperts(num_experts, hidden_size, output_size, bias)
        self.top_k = min(top_k, self.num_experts)
        self.activation = activation

        self.gate = top_k_gating(
            input_size=input_size, 
            num_experts=num_experts, 
            top_k=top_k, 
            acc_aux_loss=acc_aux_loss,
            dropout=gating_dropout,
            sample_topk=sample_topk,
            hidden_size=gating_size,
            aux_loss=aux_loss,
            gate_type=gate_type,
            )

    def extra_repr(self):
        return 'k={}'.format(self.top_k)

    def get_aux_loss_and_clear(self):
        return self.gate.get_aux_loss_and_clear()

    def compute_gate(self, moe_inp, skip_mask=None):
        top_k_indices, top_k_gates, probs = self.gate(moe_inp, skip_mask=skip_mask)
        self.batch_gates, self.batch_index, expert_size, self.index_sorted_experts =\
            compute_gating(self.top_k, probs, top_k_gates, top_k_indices)
        self.expert_size = expert_size.tolist()
        return self.gate.loss

    def forward(self, x, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        bsz, length, emb_size = x.size()
        if skip_mask is not None:
            assert x.size()[:-1] == skip_mask.size(), \
                    "Skip mask should be same shape as `x`"
            skip_mask = skip_mask.flatten()[:, None]
        x = x.reshape(-1, emb_size)
        loss = self.compute_gate(x, skip_mask)

        expert_inputs = x[self.batch_index]
        h = self.experts(expert_inputs, self.expert_size)
        h = self.activation(h)
        expert_outputs = self.output_experts(h, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros(
            (bsz * length, self.output_size),
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.output_size)
        return y, loss, self.expert_size

    def map(self, x, skip_mask=None, sample_topk=0, return_indices=False):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        if skip_mask is not None:
            assert x.size()[:-1] == skip_mask.size(), \
                    "Skip mask should be same shape as `x`"
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        loss = self.compute_gate(x, skip_mask)

        expert_inputs = x[self.batch_index]
        expert_outputs = self.experts(expert_inputs, self.expert_size)

        zeros = torch.zeros((bsz * length * self.top_k, self.head_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.index_sorted_experts, expert_outputs)
        y = y.view(bsz, length, self.top_k, -1)
        return y, loss,self.expert_size

    def reduce(self, x, multiply_by_gates=True):
        bsz, length, k, emb_size = x.size()
        x = x.reshape(-1, emb_size)

        expert_inputs = x[self.index_sorted_experts]
        expert_outputs = self.output_experts(expert_inputs, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.output_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.output_size)
        return y

class ParallelLinear(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, input, expert_size_list, weight, bias=None):
        output = ParallelLinear.forward_scriptable(input, expert_size_list, weight, bias)
        ctx.save_for_backward(input, weight, bias)
        ctx.expert_size_list = expert_size_list
        return output

    @staticmethod
    @torch.jit.script
    def forward_scriptable(input: Tensor, expert_size_list: List[int],
                           weight: Tensor, bias: Optional[Tensor]):
        output_buf: Tensor = torch.empty((input.size(0), weight.size(2)),
                                         device=input.device, dtype=input.dtype)
        num_linears = weight.size(0)

        input_list = input.split(expert_size_list, dim=0)
        output_buf_list = output_buf.split(expert_size_list)

        for i in range(num_linears):
            torch.mm(input_list[i], weight[i], out=output_buf_list[i])

        if bias is not None:
            for i in range(num_linears):
                output_buf_list[i].add_(bias[i])

        output = output_buf
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        input, weight, bias = ctx.saved_tensors
        expert_size_list = ctx.expert_size_list
        return ParallelLinear.backward_scriptable(
            grad_out, input, expert_size_list,
            weight, bias
        )

    @staticmethod
    @torch.jit.script
    def backward_scriptable(grad_out: Tensor,
                 input: Tensor, expert_size_list: List[int],
                 weight: Tensor, bias: Optional[Tensor]):
        num_linears = weight.size(0)
        input_list = input.t().split(expert_size_list, dim=1)
        grad_list = grad_out.split(expert_size_list, dim=0)

        d_input_buf = torch.empty_like(input)
        d_input_buf_list = d_input_buf.split(expert_size_list, dim=0)
        d_weight_buf = torch.empty_like(weight)

        weight_t = weight.permute(0, 2, 1)

        for i in range(num_linears):
            torch.mm(grad_list[i], weight_t[i], out=d_input_buf_list[i])
            torch.mm(input_list[i], grad_list[i], out=d_weight_buf[i])

        d_input = d_input_buf
        d_weight = d_weight_buf

        if bias is not None:
            d_bias_buf = torch.empty_like(bias)
            for i in range(num_linears):
                torch.sum(grad_list[i], dim=0, keepdim=False, out=d_bias_buf[i])
            d_bias = d_bias_buf
        else:
            d_bias = None

        return d_input, None, d_weight, d_bias


class ParallelExperts(nn.Module):
    def __init__(self, num_experts, input_size, output_size, bias=False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, input_size, output_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_experts, output_size))
        else:
            self.bias = None
        self.reset_parameters()
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size

    def extra_repr(self):
        return 'num_experts={}, input_size={}, output_size={}'.format(
            self.num_experts, self.input_size, self.output_size)

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.weight, -1. / self.weight.size(1), 1. / self.weight.size(1))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs, expert_size):
        results = ParallelLinear.apply(inputs, expert_size, self.weight, self.bias)
        return results


# @torch.jit.script
def log_gmm_posterior(z, expert_centroids):
     return (
        torch.matmul(z, expert_centroids.t())
        # - 0.5 * (
        #     torch.einsum('ni,ni->n', z, z)[:, None] +
        #     torch.einsum('ni,ni->n', expert_centroids, expert_centroids)[None, :]
        # )
     )


@torch.jit.script
def compute_gating(k: int, probs: torch.Tensor, top_k_gates: torch.Tensor, top_k_indices: torch.Tensor):
    zeros = torch.zeros_like(probs)
    gates = zeros.scatter(1, top_k_indices, 1)
    expert_size = gates.long().sum(0)
    top_k_gates = top_k_gates.flatten()
    top_k_experts = top_k_indices.flatten()
    _, index_sorted_experts = top_k_experts.sort(0)
    batch_index = index_sorted_experts.div(k, rounding_mode='trunc')
    batch_gates = top_k_gates[index_sorted_experts]
    return batch_gates, batch_index, expert_size, index_sorted_experts


class top_k_gating(nn.Module):
    def __init__(
        self,
        input_size, 
        num_experts, 
        top_k,
        acc_aux_loss=False, 
        dropout=0.1,
        hidden_size=256,
        sample_topk=0,
        aux_loss='mi',
        gate_type='mlp',
    ):
        super().__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        assert top_k <= num_experts
        self.top_k = top_k
        assert sample_topk <= top_k
        self.sample_topk = sample_topk

        self.acc_aux_loss = acc_aux_loss
        self.aux_loss = aux_loss
        self.init_aux_statistics()

        self.gate_type = gate_type
        if gate_type == 'mlp':
            self.w_gate = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_experts, bias=False)
            )
        elif gate_type == 'linear':
            self.w_gate = nn.Sequential(
                nn.Linear(input_size, num_experts, bias=False)
            )
        elif gate_type == 'gmm':
            self.w_gate = nn.Linear(input_size, hidden_size, bias=False)
            self.expert_centroids = nn.Parameter(torch.empty(num_experts, hidden_size))
            nn.init.normal_(self.expert_centroids)
            self.temperature = nn.Parameter(torch.zeros(1))
        else:
            print(gate_type)
            raise NotImplementedError

    def extra_repr(self):
        return 'k={}, num_experts={}, aux_loss={}'.format(
            self.top_k, self.num_experts, self.aux_loss)

    def init_aux_statistics(self):
        if self.aux_loss == 'mi':
            self.p_e = 0.
            self.neg_H_e_given_x = 0.
            self.count_layers = 0
        else:
            self.acc_probs = 0.
            self.acc_freq = 0.
            self.acc_lsesq = 0.
            self.acc_count = 0

    def update_aux_statistics(self, probs, logits, gates, skip_mask=None):
        if self.aux_loss == 'mi':
            log_prob = torch.log_softmax(logits, dim=-1)
            self.p_e = self.p_e + probs.mean(0)
            self.neg_H_e_given_x = self.neg_H_e_given_x + (probs * log_prob).sum() / probs.size(0)
            self.count_layers += 1
        else:
            self.acc_count = self.acc_count + logits.size(0)
            self.acc_probs = self.acc_probs + probs.sum(0)
            self.acc_freq = self.acc_freq + (gates > 0).float().sum(0)
            lsesq = torch.log(torch.exp(logits).sum(dim=-1)) ** 2
            self.acc_lsesq = self.acc_lsesq + lsesq.sum()

    def get_aux_loss_and_clear(self, eps=1e-8):
        if self.aux_loss == 'mi':
            denominator = self.count_layers 
            p_e = self.p_e / denominator
            H_e = -(p_e * (p_e + eps).log()).sum()
            neg_H_e_given_x = self.neg_H_e_given_x / denominator
            miloss = -(neg_H_e_given_x + H_e)
            loss = miloss
        else:
            switchloss =  self.num_experts * (
                F.normalize(self.acc_probs, p=1, dim=0) *
                F.normalize(self.acc_freq, p=1, dim=0)
            ).sum()
            zloss = self.acc_lsesq / self.acc_count
            loss = switchloss + 0.1 * zloss

        self.init_aux_statistics()
        return loss

    def forward(self, x, skip_mask=None):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        if self.gate_type in ['linear', 'mlp']:
            logits = self.w_gate(x)
        elif self.gate_type == 'gmm':
            z = self.w_gate(x)
            logits = log_gmm_posterior(F.normalize(z, p=2, dim=-1), F.normalize(self.expert_centroids, p=2, dim=-1)) * self.temperature.exp()

        probs = torch.softmax(logits, dim=1)
        if skip_mask is not None:
            probs = torch.masked_fill(probs, (skip_mask == 0), 0)
            logits = torch.masked_fill(logits, (skip_mask == 0), 0)

        if self.training and (self.sample_topk > 0):
            _, top_km1_indices = probs.topk(self.top_k - self.sample_topk, dim=1)
            masked_probs = probs + 1e-6
            masked_probs[torch.arange(probs.size(0)).unsqueeze(
                1), top_km1_indices] = 0
            k_indices = torch.multinomial(masked_probs, self.sample_topk)
            top_k_indices = torch.cat([top_km1_indices, k_indices], dim=-1)
            top_k_gates = torch.gather(probs, 1, top_k_indices)
        else:
            top_k_gates, top_k_indices = probs.topk(self.top_k, dim=1) 

        zeros = torch.zeros_like(probs)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        self.update_aux_statistics(probs, logits, gates, skip_mask)
        if not self.acc_aux_loss:
            self.loss = self.get_aux_loss_and_clear()
        else:
            self.loss = 0

        return top_k_indices, top_k_gates, probs