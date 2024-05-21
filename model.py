import torch.nn as nn
import torch
from transformers.modeling_outputs import ModelOutput
import torch.nn.functional as F
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
from typing import Optional
from torch import Tensor

@dataclass
class GuardrailOutput(ModelOutput):
    q_emb: Optional[Tensor] = None
    p_emb: Optional[Tensor] = None
    guardrails: Optional[Tensor] = None

class CosineNormalizerVector(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm((dim,)),
            nn.Linear(dim, 3072),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.LayerNorm((3072,)),
            nn.Linear(3072, 3072),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(3072, dim),
        )
        self.scale_factor = nn.Parameter(torch.tensor(10, dtype=torch.float64, requires_grad=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, query_input, item_input=None):
        base_output = self.layers(query_input)
        return GuardrailOutput(q_emb=query_input, p_emb=item_input, guardrails=base_output)

    
class CosineNormalizerLinearOffset(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm((dim,)),
            nn.Linear(dim, 3072),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.LayerNorm((3072,)),
            nn.Linear(3072, 3072),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(3072, 2),
        )
        self.scale_factor = nn.Parameter(torch.tensor(10, dtype=torch.float64, requires_grad=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, query_input, item_input=None):
        base_output = self.layers(query_input)
        return GuardrailOutput(q_emb=query_input, p_emb=item_input, guardrails=base_output)
        #return torch.unsqueeze(base_output[:, 0], 1), torch.unsqueeze(base_output[:, 1], 1)

class CosineNormalizerScalerOffset(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm((dim,)),
            nn.Linear(dim, 3072),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.LayerNorm((3072,)),
            nn.Linear(3072, 3072),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(3072, 1),
        )
        self.scale_factor = nn.Parameter(torch.tensor(10, dtype=torch.float64, requires_grad=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, query_input, item_input=None):
        base_output = self.layers(query_input)
        return GuardrailOutput(q_emb=query_input, p_emb=item_input, guardrails=base_output)
    
class CosineNormalizerPolyOffset(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm((dim,)),
            nn.Linear(dim, 3072),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.LayerNorm((3072,)),
            nn.Linear(3072, 3072),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(3072, 3),
        )
        self.scale_factor = nn.Parameter(torch.tensor(10, dtype=torch.float64, requires_grad=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, query_input, item_input=None):
        base_output = self.layers(query_input)
        return GuardrailOutput(q_emb=query_input, p_emb=item_input, guardrails=base_output)
        #return torch.unsqueeze(base_output[:, 0], 1), torch.unsqueeze(base_output[:, 1], 1), torch.unsqueeze(base_output[:, 2], 1)

from pytorch_lightning import LightningModule
from transformers import AutoModel, AutoTokenizer

class BaseModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
        self.model.eval()

    def forward(self, input_dict):
        outputs = self.model(**input_dict, return_dict=True)
        emb = outputs.last_hidden_state[:, 0]
        emb = F.normalize(emb, dim=-1)
        return emb        
    
