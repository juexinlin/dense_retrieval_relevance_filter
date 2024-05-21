python train.py --model_name_or_path intfloat/simlm-base-msmarco-finetuned --do_train --num_train_epochs 2 --save_strategy steps  --train_file ../unilm/simlm/data/msmarco_bm25_official/train.jsonl --validation_file ../unilm/simlm/data/msmarco_bm25_official/dev.jsonl --output_dir simlm-base-msmarco-finetuned_ls32_power2 --data_dir  ../unilm/simlm/data/msmarco_bm25_official/ --fp16 --per_device_train_batch_size 128 --train_n_passages 32 
python train.py --model_name_or_path intfloat/simlm-base-msmarco-finetuned --do_eval  --validation_file ../unilm/simlm/simlm-base-msmarco-finetuned/dev.msmarco.jsonl --output_dir simlm-base-msmarco-finetuned_ls32_power2/checkpoint-4500/  --data_dir  ../unilm/simlm/data/msmarco_bm25_official/   --train_n_passages 1000  

python train.py --model_name_or_path intfloat/simlm-base-msmarco-finetuned --do_train --num_train_epochs 2 --save_strategy steps  --train_file ../unilm/simlm/data/msmarco_bm25_official/train.jsonl --validation_file ../unilm/simlm/data/msmarco_bm25_official/dev.jsonl --output_dir simlm-base-msmarco-finetuned_poly_ls32_sigmoid --data_dir  ../unilm/simlm/data/msmarco_bm25_official/ --fp16 --per_device_train_batch_size 128 --model_type polynomial_offset --train_n_passages 32 

python train.py --model_name_or_path intfloat/simlm-base-msmarco-finetuned --do_eval  --validation_file ../unilm/simlm/simlm-base-msmarco-finetuned/dev.msmarco.jsonl --output_dir simlm-base-msmarco-finetuned_poly_ls32_sigmoid/checkpoint-4500/  --data_dir  ../unilm/simlm/data/msmarco_bm25_official/   --train_n_passages 1000 --model_type polynomial_offset

#python train.py --model_name_or_path intfloat/simlm-base-msmarco --do_eval --num_train_epochs 2 --save_strategy epoch   --validation_file ../unilm/simlm/simlm-base-msmarco-finetuned/dev.msmarco.jsonl --output_dir simlm-base-msmarco_rg_ls8/checkpoint-2000/    --data_dir  ../unilm/simlm/data/msmarco_bm25_official/   --train_n_passages 1000

#python train.py --model_name_or_path intfloat/simlm-base-msmarco --do_eval  --validation_file ../unilm/simlm/simlm-base-msmarco-finetuned/dev.msmarco.jsonl --output_dir simlm-base-msmarco_rg_ls8_power2/checkpoint-2000/  --data_dir  ../unilm/simlm/data/msmarco_bm25_official/   --train_n_passages 1000

#python train.py --model_name_or_path intfloat/simlm-base-msmarco --do_eval  --validation_file ../unilm/simlm/simlm-base-msmarco-finetuned/dev.msmarco.jsonl --output_dir simlm-base-msmarco_rg_poly_ls8_sigmoid/checkpoint-2000/  --data_dir  ../unilm/simlm/data/msmarco_bm25_official/   --train_n_passages 1000 --model_type polynomial_offset

# dummy job
#python train.py --model_name_or_path intfloat/simlm-base-msmarco --do_train --num_train_epochs 40 --save_strategy epoch  --train_file ../unilm/simlm/data/msmarco_bm25_official/train.jsonl --validation_file ../unilm/simlm/data/msmarco_bm25_official/dev.jsonl --output_dir tmp --data_dir  ../unilm/simlm/data/msmarco_bm25_official/ --fp16 --per_device_train_batch_size 512
# eval
#python train.py --model_name_or_path intfloat/simlm-base-msmarco --do_eval --num_train_epochs 2 --save_strategy epoch  --train_file ../unilm/simlm/data/msmarco_bm25_official/train.jsonl --validation_file ../unilm/simlm/data/msmarco_bm25_official/dev.jsonl --output_dir simlm-base-msmarco_rg/ --data_dir  ../unilm/simlm/data/msmarco_bm25_official/ 
# recall set inference
#python train.py --model_name_or_path intfloat/simlm-base-msmarco --do_eval --num_train_epochs 2 --save_strategy epoch   --validation_file ../unilm/simlm/simlm-base-msmarco-finetuned/dev.msmarco.jsonl --output_dir simlm-base-msmarco_rg_old/   --data_dir  ../unilm/simlm/data/msmarco_bm25_official/   --train_n_passages 1000
