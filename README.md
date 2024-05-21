# Dataset: Download pre-processed data from simlm

Details are avaialble at [simlm repo](https://github.com/microsoft/unilm/tree/master/simlm). Paper [SimLM: Pre-training with Representation Bottleneck for Dense Passage Retrieval](https://aclanthology.org/2023.acl-long.125.pdf).

The following script will download the pre-processed data for [MS-MARCO passage ranking](https://microsoft.github.io/msmarco/) task.

```shell
bash scripts/download_msmarco_data.sh
```

# Model training 
The following script is used to train the guardrail model.
```shell
MODEL_DIR=simlm-base-msmarco-finetuned_ls32
DATA_DIR=data/msmarco_bm25_official/
MODEL_NAME=intfloat/simlm-base-msmarco-finetuned

python train.py --model_name_or_path $MODEL_NAME --do_train --num_train_epochs 2 --save_strategy steps  --train_file $DATA_DIR/train.jsonl --validation_file $DATA_DIR/dev.jsonl --output_dir $MODEL_DIR --data_dir $DATA_DIR --fp16 --per_device_train_batch_size 128 --train_n_passages 32
```

# Evaluation
We perform the model evaluation on ms-marco dev set. 
1. conduct ANN retrieval for an existing tower model
2. apply guardrail model for dev queries and compute the adjusted scores
3. find a threshold that achieves recall at 99%/95%
4. apply the threshold to filter recall set and compute precision, and AUC for precision and recall curve 

## ANN retrieval for dev queries
To evaluate the performance of guardrail model, one should conduct the nearest-neighbor search on an existing retrieval model. For example, follow the instructions for [simlm biencoder retriever](https://github.com/microsoft/unilm/tree/master/simlm#evaluate-our-fine-tuned-biencoder-retriever).

## Inference
Prepare the retrieved items for inference.
```shell
OUTPUT_FILE=simlm-base-msmarco-finetuned/dev.msmarco.jsonl
RECALL_FILE=simlm-base-msmarco-finetuned/dev.msmarco.txt
python dev_recall_data_processs.py --output_file $OUTPUT_FILE --recall_file $RECALL_FILE --data_dir $DATA_DIR
```

Inference on dev recall
```shell
python train.py --model_name_or_path intfloat/simlm-base-msmarco-finetuned --do_eval  --validation_file $OUTPUT_FILE --output_dir $MODEL_DIR  --data_dir $DATA_DIR   --train_n_passages 1000
```

## Compute metrics
```shell
python metrics.py --data_dir $DATA_DIR --inference_output $MODEL_DIR/results_test.json
```
