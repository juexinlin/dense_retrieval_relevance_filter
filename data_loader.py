import os
import random

from typing import Tuple, Dict, List, Optional
from datasets import load_dataset, DatasetDict, Dataset
from transformers.file_utils import PaddingStrategy
from transformers import PreTrainedTokenizerFast, Trainer

from config import Arguments
from logger_config import logger

from typing import List, Dict


def _slice_with_mod(elements: List, offset: int, cnt: int) -> List:
    return [elements[(offset + idx) % len(elements)] for idx in range(cnt)]


def group_doc_ids(examples: Dict[str, List],
                  negative_size: int,
                  offset: int,
                  corpus_length: int,
                  use_first_positive: bool = False) -> List[int]:
    pos_doc_ids: List[int] = []
    positives: List[Dict[str, List]] = examples.get('positives', {})
    for idx, ex_pos in enumerate(positives):
        all_pos_doc_ids = ex_pos['doc_id']
        if use_first_positive:
            # keep positives that has higher score than all negatives
            all_pos_doc_ids = [doc_id for p_idx, doc_id in enumerate(all_pos_doc_ids)
                               if p_idx == 0 or ex_pos['score'][p_idx] >= ex_pos['score'][0]
                               or ex_pos['score'][p_idx] > max(examples['negatives'][idx]['score'])]

        cur_pos_doc_id = _slice_with_mod(all_pos_doc_ids, offset=offset, cnt=1)[0]
        pos_doc_ids.append(int(cur_pos_doc_id))

    if not pos_doc_ids:
        negative_size += 1 # for eval set, where no positives are provided

    neg_doc_ids: List[List[int]] = []
    negatives: List[Dict[str, List]] = examples.get('negatives', {})
    for ex_neg in negatives:
        negative_ids = [x for x in ex_neg['doc_id'] if int(x) < corpus_length]
        if len(negative_ids) < len(ex_neg['doc_id']):
            logger.info(f"truncate negative lists from {len(ex_neg['doc_id'])} to {len(negative_ids)}.")
        cur_neg_doc_ids = _slice_with_mod(negative_ids,
                                          offset=offset * negative_size,
                                          cnt=negative_size)
        cur_neg_doc_ids = [int(doc_id) for doc_id in cur_neg_doc_ids]
        neg_doc_ids.append(cur_neg_doc_ids)

    #assert len(pos_doc_ids) == len(neg_doc_ids), '{} != {}'.format(len(pos_doc_ids), len(neg_doc_ids))
    #assert all(len(doc_ids) == negative_size for doc_ids in neg_doc_ids)

    input_doc_ids: List[int] = []
    if len(pos_doc_ids) == len(neg_doc_ids):
        for pos_doc_id, neg_ids in zip(pos_doc_ids, neg_doc_ids):
            input_doc_ids.append(pos_doc_id)
            input_doc_ids += neg_ids
    else:
        doc_ids = neg_doc_ids if neg_doc_ids else pos_doc_ids
        for doc_id in doc_ids:
            input_doc_ids += doc_id

    return input_doc_ids


class RetrievalDataLoader:

    def __init__(self, args: Arguments, tokenizer: PreTrainedTokenizerFast):
        self.args = args
        self.negative_size = args.train_n_passages - 1
        assert self.negative_size > 0
        self.tokenizer = tokenizer
        corpus_path = os.path.join(args.data_dir, 'passages.jsonl.gz')
        self.corpus: Dataset = load_dataset('json', data_files=corpus_path)['train']
        self.train_dataset, self.eval_dataset = self._get_transformed_datasets()

        # use its state to decide which positives/negatives to sample
        self.trainer: Optional[Trainer] = None

    def _transform_func(self, examples: Dict[str, List]) -> Dict[str, List]:
        if self.trainer:
            current_epoch = int(self.trainer.state.epoch or 0)
        else:
            current_epoch = 0

        input_doc_ids: List[int] = group_doc_ids(
            examples=examples,
            negative_size=self.negative_size,
            offset=current_epoch + self.args.seed,
            corpus_length=len(self.corpus),
            use_first_positive=self.args.use_first_positive
        )
        assert len(input_doc_ids) == len(examples['query']) * self.args.train_n_passages

        input_docs: List[str] = [self.corpus[doc_id]['contents'] for doc_id in input_doc_ids]
        input_titles: List[str] = [self.corpus[doc_id]['title'] for doc_id in input_doc_ids]

        query_batch_dict = self.tokenizer(examples['query'],
                                          max_length=self.args.q_max_len,
                                          padding=PaddingStrategy.DO_NOT_PAD,
                                          truncation=True)
        doc_batch_dict = self.tokenizer(input_titles,
                                        text_pair=input_docs,
                                        max_length=self.args.p_max_len,
                                        padding=PaddingStrategy.DO_NOT_PAD,
                                        truncation=True)

        merged_dict = {k: v for k, v in query_batch_dict.items()}
        step_size = self.args.train_n_passages
        for k, v in doc_batch_dict.items():
            item_name = 'd_{}'.format(k)
            merged_dict[item_name] = []
            for idx in range(0, len(v), step_size):
                merged_dict[item_name].append(v[idx:(idx + step_size)])

        label_name = 'labels'
        merged_dict[label_name] = []
        if 'positives' in examples: 
            # train set / dev set
            for idx in range(0, len(input_doc_ids), step_size):
                merged_dict[label_name].append([1] + [0] * (step_size - 1))
        else:
            # inference for recall set
            merged_dict['doc_ids'] = []
            for x in examples['negatives']:
                merged_dict[label_name].append(x['score'])
                merged_dict['doc_ids'].append(x['doc_id'])
            merged_dict['query_ids'] = []
            for x in examples['query_id']:
                merged_dict['query_ids'].append([x] * step_size)
            

        # Custom formatting function must return a dict
        return merged_dict

    def _get_transformed_datasets(self) -> Tuple:
        data_files = {}
        if self.args.train_file is not None:
            data_files["train"] = self.args.train_file.split(',')
        if self.args.validation_file is not None:
            data_files["validation"] = self.args.validation_file
        raw_datasets: DatasetDict = load_dataset('json', data_files=data_files)

        train_dataset, eval_dataset = None, None

        if self.args.do_train:
            if "train" not in raw_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = raw_datasets["train"]
            if self.args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(self.args.max_train_samples))
            # Log a few random samples from the training set:
            for index in random.sample(range(len(train_dataset)), 3):
                logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
            train_dataset.set_transform(self._transform_func)

        if self.args.do_eval:
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["validation"]
            eval_dataset.set_transform(self._transform_func)

        return train_dataset, eval_dataset
