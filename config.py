import os
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments


from logger_config import logger


@dataclass
class Arguments(TrainingArguments):
    model_name_or_path: str = field(
        default='bert-base-uncased',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    data_dir: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )
        
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics on (a jsonlines file)."
        },
    )

    train_n_passages: int = field(
        default=2,
        metadata={"help": "number of passages for each example (including both positive and negative passages)"}
    )
    share_encoder: bool = field(
        default=True,
        metadata={"help": "no weight sharing between qry passage encoders"}
    )
    use_first_positive: bool = field(
        default=False,
        metadata={"help": "Always use the first positive passage"}
    )
    include_inputs_for_metrics: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    save_total_limit: int = field(default=3)
    metric_for_best_model: str = field(default='auc_improvement')
    greater_is_better: bool = field(default=True)
    model_type: str = field(default='linear_offset')

    # following arguments are used for encoding documents
    do_encode: bool = field(default=False, metadata={"help": "run the encoding loop"})
    encode_in_path: str = field(default=None, metadata={"help": "Path to data to encode"})
    encode_save_dir: str = field(default=None, metadata={"help": "where to save the encode"})
    encode_shard_size: int = field(default=int(2 * 10**6))
    encode_batch_size: int = field(default=256)

    # used for index search
    do_search: bool = field(default=False, metadata={"help": "run the index search loop"})
    search_split: str = field(default='dev', metadata={"help": "which split to search"})
    search_batch_size: int = field(default=128, metadata={"help": "query batch size for index search"})
    search_topk: int = field(default=200, metadata={"help": "return topk search results"})
    search_out_dir: str = field(default='', metadata={"help": "output directory for writing search results"})

    q_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query."
        },
    )
    p_max_len: int = field(
        default=144,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    dry_run: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set dry_run to True for debugging purpose'}
    )

    def __post_init__(self):
        assert os.path.exists(self.data_dir)
        assert torch.cuda.is_available(), 'Only support running on GPUs'

        if self.dry_run:
            self.logging_steps = 1
            self.max_train_samples = self.max_train_samples or 128
            self.num_train_epochs = 1
            self.per_device_train_batch_size = min(2, self.per_device_train_batch_size)
            self.train_n_passages = min(4, self.train_n_passages)
            self.gradient_accumulation_steps = 1
            self.max_steps = 30
            self.save_steps = self.eval_steps = 30
            logger.warning('Dry run: set logging_steps=1')

        if self.do_encode:
            assert self.encode_save_dir
            os.makedirs(self.encode_save_dir, exist_ok=True)
            assert os.path.exists(self.encode_in_path)

        if torch.cuda.device_count() <= 1:
            self.logging_steps = min(10, self.logging_steps)

        super(Arguments, self).__post_init__()

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        self.label_names = ['labels']
