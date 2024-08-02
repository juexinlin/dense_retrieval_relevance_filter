import argparse
from transformers import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from collator import BiencoderCollator
from config import Arguments
from data_loader import RetrievalDataLoader
from logger_config import logger, LoggerCallback
import logging
import torch
from transformers import HfArgumentParser, AutoTokenizer, AutoModel
import inspect
from transformers.trainer_callback import PrinterCallback
from guardrail_trainer import GuardrailTrainer, compute_adjusted_score, _unpack_qp
from model import CosineNormalizerVector, CosineNormalizerLinearOffset, CosineNormalizerScalerOffset, CosineNormalizerPolyOffset, BaseModel, Choppy
from guardrail_trainer import dot_product
import os
from functools import partial
from torcheval.metrics.functional import binary_auprc
import json
import tqdm
import numpy as np
from collections import Counter

fileHandler = logging.FileHandler(filename='training_info.log', mode='w')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(fileHandler)

def filter_arguments(mydict,my_class):
    filtered_mydict = {k: v for k, v in mydict.items() if
                       k in [p.name for p in inspect.signature(my_class.__init__).parameters.values()]}
    return filtered_mydict

def get_model(model_type, input_dim,train_n_passages):
    if model_type == 'vector':
        model = CosineNormalizerVector(input_dim)
    elif model_type == 'scaler_offset':
        model = CosineNormalizerScalerOffset(input_dim)
    elif model_type == 'linear_offset':
        model = CosineNormalizerLinearOffset(input_dim)
    elif model_type == 'polynomial_offset':
        model = CosineNormalizerPolyOffset(input_dim)
    elif model_type == 'choppy':
        model = Choppy(seq_len=train_n_passages)
    else:
        raise Exception(f'Model type: {model_type} not supported')
    return model

def compute_metrics(tower_model, model_type, eval_pred):
    predictions, labels = eval_pred
    q_emb = predictions[0]
    p_emb = predictions[1]
    guardrails = predictions[2]
    adjusted_scores, model_scores = compute_adjusted_score(q_emb, p_emb, guardrails, model_type, return_model_scores=True)
    auc_improvement = binary_auprc(adjusted_scores, labels) - binary_auprc(model_scores, labels)
    return {"auc_improvement": auc_improvement}

def main():
    parser = HfArgumentParser((Arguments,))
    args = parser.parse_args_into_dataclasses()[0]

    #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)
    
    logger.info('Args={}'.format(str(args)))
    #tower_model = AutoModel.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16) 
    tower_model = BaseModel(args.model_name_or_path)
    # freeze the tower model
    for name, param in tower_model.named_parameters():
        param.requires_grad = False 
    model = get_model(args.model_type, tower_model.model.config.hidden_size,args.train_n_passages)
    if args.do_eval:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, "pytorch_model.bin")))
        model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, TOKENIZERS_PARALLELISM = False)
    logger.info(model)
    logger.info('Vocab size: {}'.format(len(tokenizer)))

    data_collator = BiencoderCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if args.fp16 else None)

    retrieval_data_loader = RetrievalDataLoader(args=args, tokenizer=tokenizer)
    train_dataset = retrieval_data_loader.train_dataset
    eval_dataset = retrieval_data_loader.eval_dataset

    trainer = GuardrailTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
        tower_model=tower_model,
        model_type=args.model_type,
        compute_metrics=partial(compute_metrics, tower_model, args.model_type),
    )
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(LoggerCallback)
    retrieval_data_loader.trainer = trainer

    if args.do_train:
        train_result = trainer.train()
        trainer.save_model()

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    if args.do_eval:
        logger.info("*** Evaluate ***")
        from torch.utils.data import Dataset, DataLoader
        device = torch.device('cuda:0')
        loader = DataLoader(eval_dataset, batch_size=2, collate_fn=lambda x: {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data_collator(x).items()})
        run_recall_inference = not train_dataset # recall set inference
        if run_recall_inference: 
            f = open(os.path.join(args.output_dir, 'results_test.json'), 'w')
        adjusted_scores_list = []
        model_scores_list = []
        label_list = []
        argmax_predictions=[]
        retrieved_positives = 0
        all_positives = 0

        for i, inputs in enumerate(tqdm.tqdm(loader, total=len(loader), desc="Processing", ncols=100)):
            query_input_dict, passage_input_dict = _unpack_qp(inputs)
            tower_model.to(device=query_input_dict['input_ids'].device)
            with torch.no_grad():
                query_emb = tower_model.forward(query_input_dict).to(torch.float32)
                passage_emb = tower_model.forward(passage_input_dict).to(torch.float32)

            model.to(device=query_input_dict['input_ids'].device)
            ### if non choppy
            if args.model_type=='choppy':
                model_scores = dot_product(query_emb, passage_emb)

                model_scores = model_scores.reshape(query_emb.shape[0], -1, 1)
                # reshape the labels in the same way
                labels = inputs['labels'].reshape(query_emb.shape[0], -1, 1)
                sorted_model_scores, indices = torch.sort(model_scores, dim=1, descending=True)
                sorted_labels = torch.gather(labels, 1, indices)

                # pass scores and labels to choppy
                sliced_tensor = sorted_model_scores[:, :model.seq_len, :]
                cuttoff_probabilities = model(sliced_tensor)
                # we compute the argmax and return that
                argmaxes = torch.argmax(cuttoff_probabilities, dim=1, keepdim=True)

                sliced_labels = []
                vectors_sizes = 0
                for i in range(sorted_labels.size(0)):
                    sliced = sorted_labels[i, :argmaxes[i] + 1]
                    vectors_sizes += sliced.size()[0]
                    sliced_labels.append(sorted_labels[i, :argmaxes[i] + 1])

                sliced_labels = torch.stack(sliced_labels)

                retrieved_positives += sliced_labels.sum().item()
                all_positives += labels.sum().item()
                argmax_predictions.extend(argmaxes.cpu().numpy().ravel().tolist())
            else:

                model_output = model(query_emb.float(), passage_emb.float())
                guardrails = model_output.guardrails
                adjusted_scores, model_scores = compute_adjusted_score(query_emb, passage_emb, model_output.guardrails, args.model_type, return_model_scores=True)

                adjusted_scores_list.extend(adjusted_scores.squeeze().tolist())
                model_scores_list.extend(model_scores.squeeze().tolist())
                label_list.extend(inputs['labels'].squeeze().tolist())

            #import pdb; pdb.set_trace()
            #binary_auprc(adjusted_scores.squeeze(), inputs['labels'].squeeze()) - binary_auprc(model_scores.squeeze(), inputs['labels'].squeeze())
            if run_recall_inference:
                for pos, label in enumerate(inputs['labels']):
                    out = {'query_id': inputs['query_ids'][pos], 'doc_id': inputs['doc_ids'][pos], 'label': float(label), 'model_score': float(model_scores[pos]), 'adjusted_score': float(adjusted_scores[pos])}
                    f.write(json.dumps(out) + '\n')

        if run_recall_inference:
            f.close()

        if args.model_type == 'choppy':
            recall  = np.round(retrieved_positives/all_positives,3)
            precision  = np.round(retrieved_positives/vectors_sizes,3)
            Counter(argmax_predictions)
            print('recall: ', recall)
            print('precision ', precision)
            print(f'argmax distribution: {argmax_predictions}')


        else:
            baseline_auc = binary_auprc(torch.tensor(model_scores_list), torch.tensor(label_list))
            adjusted_auc = binary_auprc(torch.tensor(adjusted_scores_list), torch.tensor(label_list))
            print('baseline AUC: ', baseline_auc)
            print('adjusted AUC: ', adjusted_auc)
            print('AUC improvement: ', adjusted_auc - baseline_auc)


    return


if __name__ == '__main__':
    main()
