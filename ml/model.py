"""
ml.model
------------------------------------------------------------
* DynamicMLP  – 支持任意隐藏层结构与激活
* build_model()  – 根据超参 dict 构造模型
"""
from typing import Sequence, Dict
import torch
import torch.nn as nn

_ACT = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}

class DynamicMLP(nn.Module):
    def __init__(self, in_dim: int,
                 hidden_layers: Sequence[int],
                 activation: str = "relu"):
        super().__init__()
        act_cls = _ACT.get(activation, nn.ReLU)
        layers  = []
        prev = in_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), act_cls()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (N, in) → (N,)
        return self.net(x).squeeze(1)

# ----------------------------------------------------------------------
def build_model(hparams: Dict, input_dim: int) -> nn.Module:
    """
    hparams 形如：
        { 'hidden_layers': (128,64,32),
          'activation'  : 'relu' }
    """
    return DynamicMLP(
        in_dim        = input_dim,
        hidden_layers = hparams.get("hidden_layers", (64, 32)),
        activation    = hparams.get("activation", "relu")
    )
