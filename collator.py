import torch

from dataclasses import dataclass
from typing import List, Dict, Any
from transformers import DataCollatorWithPadding, BatchEncoding


def _unpack_doc_values(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    doc_examples = []
    for f in features:
        keys = list(f.keys())
        lists_per_key = len(f[keys[0]])
        for idx in range(lists_per_key):
            doc_examples.append({k: f[k][idx] for k in keys})
    return doc_examples


@dataclass
class BiencoderCollator(DataCollatorWithPadding):

    def __call__(self, features: List[Dict[str, Any]]) -> BatchEncoding:
        d_prefix = 'd_'
        query_fields = [k.replace(d_prefix, '')  for k, v in features[0].items() if k.startswith(d_prefix)]
        query_examples = [{k: v for k, v in f.items() if k in query_fields} for f in features]
        doc_examples = _unpack_doc_values(
            [{k[len(d_prefix):]: v for k, v in f.items() if k.startswith(d_prefix)} for f in features])
        assert len(doc_examples) % len(query_examples) == 0, \
            '{} doc and {} queries'.format(len(doc_examples), len(query_examples))

        # already truncated during tokenization
        q_collated = self.tokenizer.pad(
            query_examples,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors)
        d_collated = self.tokenizer.pad(
            doc_examples,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors)

        # merge into a single BatchEncoding by adding prefix
        for k in d_collated:
            q_collated[d_prefix + k] = d_collated[k]

        merged_batch_dict = q_collated

        merged_batch_dict['labels'] = torch.tensor([f['labels'] for f in features]).view(-1, 1)

        if 'query_ids' in features[0]:
            merged_batch_dict['query_ids'] = []
            merged_batch_dict['doc_ids'] = []
            for f in features:
                merged_batch_dict['query_ids'].extend(f['query_ids'])
                merged_batch_dict['doc_ids'].extend(f['doc_ids'])

        return merged_batch_dict
