import json
import argparse
import os

def read_queries(path):
    query_dict = dict()
    with open(path, 'r') as f:
        for line in f:
            q_id, q = line.strip().split('\t')
            query_dict[q_id] = q
    return query_dict

def read_labels(path):
    relevant_dict = dict()
    with open(path, 'r') as f:
        for line in f:
            q_id, _, p_id, score = line.strip().split('\t')
            if q_id not in relevant_dict:
                relevant_dict[q_id] = [p_id]
            else:
                relevant_dict[q_id].append(p_id)
    return relevant_dict

def get_label(q_id, p_id,relevant_dict):
    if p_id in relevant_dict[q_id]:
        label = 1.
    else:
        label = 0.
    return label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file",
                        help="path to the output file",
                        default=None)
    parser.add_argument("--recall_file",
                        help="path to the recall output",
                        default=None)
    parser.add_argument("--data_dir",
                        help="path to train directory",
                        default=None)
    args = parser.parse_args()

    query_dict = read_queries(os.path.join(args.data_dir, 'dev_queries.tsv'))
    relevant_dict = read_labels(os.path.join(args.data_dir, 'dev_qrels.txt'))

    f_out = open(args.output_file, 'w')

    with open(args.recall_file, 'r') as f:
        out = {}
        for line in f:
            q_id, p_id, rank, score = line.strip().split('\t')
            if not out:
                if q_id not in query_dict:
                    continue
                out['query_id'] = q_id
                out['query'] = query_dict[q_id]
                label = get_label(q_id, p_id,relevant_dict)
                out['negatives'] = {"doc_id": [p_id], 'score': [label]}
            elif out and out['query_id'] == q_id:
                out['negatives']['doc_id'].append(p_id)
                out['negatives']['score'].append(get_label(q_id, p_id,relevant_dict))
            elif out and out['query_id'] != q_id:
                f_out.write(json.dumps(out) + '\n')
                if q_id in query_dict:
                    out = {'query_id': q_id, 'query': query_dict[q_id],
                           'negatives': {"doc_id": [p_id], 'score': [get_label(q_id, p_id,relevant_dict)]}}
    if out:
        f_out.write(json.dumps(out) + '\n')
    f_out.close()
    return


if __name__ == '__main__':
    """
    OUTPUT_FILE=simlm-base-msmarco-finetuned/dev.msmarco.jsonl
    RECALL_FILE=simlm-base-msmarco-finetuned/dev.msmarco.txt
    python dev_recall_data_processs.py --output_file $OUTPUT_FILE --recall_file $RECALL_FILE --data_dir $DATA_DIR

    sample output:
    {"query_id": "1000000", "query": "where does real insulin come from", "negatives": {"doc_id": ["961686", "7662409", ...], "score": [0.0, 1.0, ...]}}
    """
    main()

