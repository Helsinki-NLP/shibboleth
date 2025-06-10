#!/usr/bin/env python
# coding: utf-8

import captum
from captum.attr import LayerIntegratedGradients, TokenReferenceBase
import pandas as pd
import torch
import tqdm
import transformers

import argparse
import gc
import collections
import json



def main(args):
    data = pd.read_csv(args.dataset_path, sep="\t", names=['labels', 'raw', 'tok'], quoting=3)
    label2id, id2label = {}, {}
    with open(f"{args.checkpoint_path}/../labels.json") as labelfile:
        label2id = json.load(labelfile)
        id2label = {label2id[label]: label for label in label2id}
    data["labels"] = data["labels"].map(label2id)

    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.checkpoint_path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(args.checkpoint_path).eval().to(device)
    pad_idx = tokenizer.pad_token_id
    lig = LayerIntegratedGradients(lambda ids: model.to(ids.device)(input_ids=ids).logits.softmax(-1), model.get_input_embeddings())
    token_reference = TokenReferenceBase(reference_token_idx=pad_idx)
    
    with tqdm.trange(len(data)) as pbar, open(args.output_path, 'w') as ostr:
        def ig_for_instance(row):
            model.zero_grad()
            input_text = row[args.instance_type]
            true_label = row['labels']
            device_ = device
            inputs = tokenizer(input_text, return_tensors='pt')
            if inputs.input_ids.numel() > 200: # a value of 176 seems to work with n_steps at 250
                tqdm.tqdm.write(f'input is too large for GPU ({inputs.input_ids.numel()}) — offloading this datapoint to CPU, this will be slow.')
                device_ = 'cpu'
                
            inputs = inputs.to(device_)

            seq_length = inputs.input_ids.shape[-1]

            # predict & populate gradient tensors
            pred = model.to(device_)(**inputs)
            pred_ind = pred.logits.softmax(-1).argmax().item()

            # generate reference indices (= baseline data) for each sample
            reference_indices = token_reference.generate_reference(seq_length, device=device_).unsqueeze(0)

            # compute attributions and approximation delta using layer integrated gradients
            attributions_ig, delta = lig.attribute(inputs.input_ids, reference_indices, \
                                                   n_steps=args.n_steps_ig, return_convergence_delta=True, target=pred_ind)

            # aggregate across embedding dimensions per word
            attributions = attributions_ig.sum(dim=2).squeeze(0)
            # normalize across sentence
            attributions = attributions / torch.norm(attributions)
            attributions = attributions.tolist()

            word_ids = torch.tensor([[-1 if idx is None else idx for idx in inputs.word_ids()]], device=device_)
            word_id_to_str = [
                tokenizer.decode(inputs.input_ids.masked_select(word_ids == idx))
                for idx in range(word_ids.max() + 1)
            ]
            assert len(inputs.word_ids()) == len(attributions)
            attribs_parsed = [0.0 for _ in range(word_ids.max() + 1)]
            tokens_parsed = word_id_to_str
            for word_id, ig_score in zip(inputs.word_ids(), attributions):
                if word_id is None: continue
                attribs_parsed[word_id] += ig_score
            assert len(attribs_parsed) == len(tokens_parsed)
            attributions_dict = {
                'correct': pred_ind == true_label,
                'pred_label': id2label[pred_ind],
                'gold_label': id2label[true_label],
                'tokens': tokens_parsed,
                'attribs': attribs_parsed,
            }
            torch.cuda.empty_cache()
            gc.collect()
            pbar.update()
            return attributions_dict


        for attrib_dict in data.apply(ig_for_instance, axis=1).to_list():
            print(json.dumps(attrib_dict, ensure_ascii=False), file=ostr)

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('checkpoint_path')
    cli_parser.add_argument('dataset_path')
    cli_parser.add_argument('output_path')
    cli_parser.add_argument('instance_type', choices=['raw', 'tok'])
    cli_parser.add_argument('--n_steps_ig', default=100, type=int)
    # cli_parser.add_argument('--agg_max', action='store_true', help='use max attribution rather than sum to aggregate per type')
    # cli_parser.add_argument('--ml', '-ml', action='store_true')
    args = cli_parser.parse_args()
    main(args)


