#!/usr/bin/env python
# coding: utf-8

import lime.lime_text
import pandas as pd
import torch
import tqdm
import transformers
import datasets
import numpy as np

import argparse
import gc
import json
import collections



@torch.no_grad()
def main(args):
    data = pd.read_csv(args.dataset_path, sep="\t", names=['labels', 'raw', 'tok'], quoting=3)
    labels = sorted(data['labels'].unique())
    id2label = {idx:label for idx, label in enumerate(labels)}
    label2id = {label:idx for idx, label in enumerate(labels)}
    data["labels"] = data["labels"].map(label2id)

    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.checkpoint_path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(args.checkpoint_path).eval().to(device)

    # build a pipeline object to do predictions
    predictor = transformers.pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0,
        return_all_scores=True,
        batch_size=32,
    )
    data['predicted_labels'] = predictor(data[args.instance_type].to_list())
    data['predicted_labels'] = data['predicted_labels'].apply(
        lambda lst: max(lst, key=lambda item:item['score'])['label']
    ).map({f'LABEL_{i}': i for i in range(len(labels))})
    data['correct'] = data['predicted_labels'] == data['labels']

    def split_into_words(text):
        encoding = tokenizer(text, return_tensors='pt')
        word_ids = torch.tensor([[-1 if idx is None else idx for idx in encoding.word_ids()]])
        word_id_to_str = [
            tokenizer.decode(encoding.input_ids.masked_select(word_ids == idx))
            for idx in range(word_ids.max() + 1)
        ]
        return word_id_to_str


    explainer = lime.lime_text.LimeTextExplainer(
        class_names=labels,
        # split_expression=split_into_words,  # unfortunately that seems to not work
        feature_selection='none',
    )

    def lime_comp_pred(texts):
        return np.array([
            [p['score'] for p in sorted(pred, key=lambda item: int(item['label'].split('_')[1]))]
            for pred in predictor(texts)
        ])

    with open(args.output_path, 'w') as ostr, tqdm.trange(len(data)) as pbar:

        def to_attrib_dict(row):
            lime_values = explainer.explain_instance(
                row[args.instance_type],
                lime_comp_pred,
                labels=(row['predicted_labels'],),
            )
            lime_values = lime_values.as_list(row['predicted_labels'])

            tokens, attribs = zip(*lime_values)
            attrib_dict = {
                 'correct': row["correct"],
                 'pred_label': id2label[row["predicted_labels"]],
                 'gold_label': id2label[row["labels"]],
                 'tokens': tokens,
                 'attribs': attribs,
            }
            pbar.update()
            return attrib_dict

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


