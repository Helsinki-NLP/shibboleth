#!/usr/bin/env python
# coding: utf-8

import shap
import pandas as pd
import torch
import tqdm
import transformers
import datasets

import argparse
import gc
import json
import collections



@torch.no_grad()
def main(args):
    data = pd.read_csv(args.dataset_path, sep="\t", names=['labels', 'raw', 'tok'], quoting=3)
    labels = sorted(data['labels'].unique())
    label2id, id2label = {}, {}
    with open(f"{args.checkpoint_path}/../labels.json") as labelfile:
        label2id = json.load(labelfile)
        id2label = {label2id[label]: label for label in label2id}
    data["labels"] = data["labels"].map(label2id)

    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.checkpoint_path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(args.checkpoint_path).eval().to(device)

    # build a pipeline object to do predictions
    pred = transformers.pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0,
        top_k=None,
        batch_size=32,
    )
    data['prediction_scores'] = pred(data[args.instance_type].to_list())
    data['predicted_labels'] = data['prediction_scores'].apply(
        lambda lst: max(lst, key=lambda item:item['score'])['label']
    ).map({f'LABEL_{i}': i for i in range(len(labels))})
    data['correct'] = data['predicted_labels'] == data['labels']

    explainer = shap.Explainer(pred)
    shap_values = explainer(data[args.instance_type])
    data['shap_values'] =  shap_values.values
    data['shap_data'] =  shap_values.data

    with open(args.output_path, 'w') as ostr:
        def remap_shaps_to_words(shap_data, shap_values, text, label):
            encoding = tokenizer(text, return_tensors='pt')
            word_ids = torch.tensor([[-1 if idx is None else idx for idx in encoding.word_ids()]])
            word_id_to_str = [
               tokenizer.decode(encoding.input_ids.masked_select(word_ids == idx))
                for idx in range(word_ids.max() + 1)
            ]
            assert len(shap_data) ==  len(encoding.word_ids())
            attributions = shap_values[...,label]
            attribs_parsed = [0.0 for _ in range(word_ids.max() + 1)]
            tokens_parsed = word_id_to_str
            for word_id, shap_score in zip(encoding.word_ids(), attributions):
                if word_id is None: continue
                attribs_parsed[word_id] += shap_score
            assert len(attribs_parsed) == len(tokens_parsed)
            return {'tokens': tokens_parsed, 'attribs': attribs_parsed}

        def to_attrib_dict(row):
            attribs = remap_shaps_to_words(
                 row["shap_data"],
                 row["shap_values"],
                 row[args.instance_type],
                 row["predicted_labels"],
            )
            return {
                 'correct': row["correct"],
                 'pred_label': id2label[row["predicted_labels"]],
                 'gold_label': id2label[row["labels"]],
                 'pred_scores': {id2label[int(item["label"].replace("LABEL_", ""))]: item["score"] for item in row["prediction_scores"]},
                 **attribs,
            }

        for attrib_dict in data.apply(to_attrib_dict, axis=1).to_list():
            print(json.dumps(attrib_dict, ensure_ascii=False), file=ostr)

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('checkpoint_path')
    cli_parser.add_argument('dataset_path')
    cli_parser.add_argument('output_path')
    cli_parser.add_argument('instance_type', choices=['raw', 'tok'])
    cli_parser.add_argument('--n_steps_ig', default=100, type=int)
    # cli_parser.add_argument('--ml', '-ml', action='store_true')
    # cli_parser.add_argument('--agg_max', action='store_true', help='use max attribution rather than sum to aggregate per type')
    args = cli_parser.parse_args()
    main(args)


