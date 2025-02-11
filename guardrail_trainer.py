from torch import nn
from transformers import Trainer
from typing import Tuple, Dict, List, Optional
import torch
import torch.nn.functional as F

def _unpack_qp(inputs: Dict[str, torch.Tensor]) -> Tuple:
    d_prefix = 'd_' 
    doc_batch_dict = {k[len(d_prefix):]: v for k, v in inputs.items() if k.startswith(d_prefix)}
    query_batch_dict = {k: v for k, v in inputs.items() if k in doc_batch_dict}

    if not query_batch_dict:
        query_batch_dict = None
    if not doc_batch_dict:
        doc_batch_dict = None

    return query_batch_dict, doc_batch_dict

def dot_product(query_emb, passage_emb):
    """
    input
    query_emb: batch_size x emb_size
    passage_emb: (batch_size * train_n_passages) x emb_size
    return
    scores: (batch_size * train_n_passages) x 1
    """
    train_n_passages = int(passage_emb.shape[0] / query_emb.shape[0])
    query_emb_expanded = query_emb.repeat_interleave(train_n_passages, dim=0)
    scores = (passage_emb * query_emb_expanded).sum(dim=-1, keepdim=True) # dot product for each pair
    return scores

def encode(tower_model, input_dicts):
    outputs = tower_model(**input_dicts, return_dict=True)
    emb = outputs.last_hidden_state[:, 0]
    emb = F.normalize(emb, dim=-1)
    return emb


def compute_adjusted_score(query_emb, passage_emb, guardrails, model_type, return_model_scores=False):
    model_scores = dot_product(query_emb, passage_emb)
    if model_type == 'vector':
        adjusted_scores = dot_product(guardrails, passage_emb)
    elif model_type == 'scaler_offset':
        adjusted_scores = model_scores + guardrails
    elif model_type == 'linear_offset':
        train_n_passages = int(passage_emb.shape[0] / query_emb.shape[0])
        param1 = torch.unsqueeze(guardrails[:, 0], 1)
        param2 = torch.unsqueeze(guardrails[:, 1], 1)
        param1 = param1.repeat_interleave(train_n_passages, dim=0)
        param2 = param2.repeat_interleave(train_n_passages, dim=0)
        adjusted_scores = param1 * model_scores + param2
    elif model_type == 'sqrt_offset':
        train_n_passages = int(passage_emb.shape[0] / query_emb.shape[0])
        param1 = torch.unsqueeze(guardrails[:, 0], 1)
        param2 = torch.unsqueeze(guardrails[:, 1], 1)
        param1 = param1.repeat_interleave(train_n_passages, dim=0)
        param2 = param2.repeat_interleave(train_n_passages, dim=0)
        adjusted_scores = param1 * model_scores**0.5 + param2
    elif model_type == 'quadratic_offset':
        train_n_passages = int(passage_emb.shape[0] / query_emb.shape[0])
        param1 = torch.unsqueeze(guardrails[:, 0], 1)
        param2 = torch.unsqueeze(guardrails[:, 1], 1)
        param1 = param1.repeat_interleave(train_n_passages, dim=0)
        param2 = param2.repeat_interleave(train_n_passages, dim=0)
        adjusted_scores = param1 * model_scores**2 + param2
    elif model_type == 'polynomial_offset':
        train_n_passages = int(passage_emb.shape[0] / query_emb.shape[0])
        param1 = torch.unsqueeze(guardrails[:, 0], 1)
        param2 = torch.unsqueeze(guardrails[:, 1], 1)
        param3 = torch.unsqueeze(guardrails[:, 2], 1)
        param1 = param1.repeat_interleave(train_n_passages, dim=0)
        param2 = param2.repeat_interleave(train_n_passages, dim=0)
        param3 = param3.repeat_interleave(train_n_passages, dim=0)
        adjusted_scores = param1 * model_scores**(2*torch.sigmoid(param3)) + param2
    else:
        raise Exception(f'Model type: {model_type} not supported')
    if return_model_scores:
        return adjusted_scores, model_scores
    else:
        return adjusted_scores



class GuardrailTrainer(Trainer):
    def __init__(self, model_type, tower_model, *args, **kwargs):
        self.model_type = model_type
        self.tower_model = tower_model
        super(GuardrailTrainer, self).__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        query_input_dict, passage_input_dict = _unpack_qp(inputs)
        self.tower_model.to(device=query_input_dict['input_ids'].device)
        with torch.no_grad():
            query_emb = self.tower_model.forward(query_input_dict)
            passage_emb = self.tower_model.forward(passage_input_dict)
        model_output = model(query_emb, passage_emb)
        guardrails = model_output.guardrails
        adjusted_scores = compute_adjusted_score(query_emb, passage_emb, model_output.guardrails, self.model_type)
        predicted_scores = adjusted_scores
        loss = nn.BCEWithLogitsLoss(reduction='mean')(predicted_scores, inputs['labels'].float())
        return (loss, guardrails) if return_outputs else loss
