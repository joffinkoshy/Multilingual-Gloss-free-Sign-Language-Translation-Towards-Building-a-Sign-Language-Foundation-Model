"""Proposed novelty — LID self-conditioning (PLAN_LID_SELF_CONDITIONING.md).

Sign2(LID+Text) computes the token-level LID-CTC posterior Z = softmax(W_lid
h_int) from the first encoder layer but only uses it as a loss target. This
module feeds Z back into the residual stream before the Sign2Text layers
(self-conditioned CTC, Nozaki & Komatsu, Interspeech 2021):

    h_int' = h_int + g * W_c Z,   g = sigmoid(a)

W_c is a bias-free Linear(V_lid -> d_model) and `a` a scalar gate logit. With
g -> 0 the model is exactly the paper's Sign2(LID+Text). Composed into
Sign2LIDText only via its `lid_conditioner=` constructor arg (built in
src/models/build.py from config.lid_self_conditioning) — the reproduction
file never special-cases this logic itself.
"""
import torch
from torch import nn


class LIDSelfConditioner(nn.Module):
    def __init__(self, lid_vocab_size: int, d_model: int, gate_init: float = -2.0, learn_gate: bool = True):
        super().__init__()
        self.proj = nn.Linear(lid_vocab_size, d_model, bias=False)
        gate_logit = torch.tensor(float(gate_init))
        if learn_gate:
            self.gate_logit = nn.Parameter(gate_logit)
        else:
            self.register_buffer("gate_logit", gate_logit)

    @property
    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def forward(self, h_int: torch.Tensor, lid_log_probs: torch.Tensor) -> torch.Tensor:
        # lid_log_probs: [B, |F|, V_lid] from the LID-CTC head. Gradient flows
        # through the posterior, so L_Txt / L_Attn also train the LID head.
        return h_int + self.gate * self.proj(lid_log_probs.exp())
