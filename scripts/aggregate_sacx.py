import argparse
import collections
import json
import string
import math
import itertools

import pandas as pd
import tqdm

import pprint

parser = argparse.ArgumentParser()
parser.add_argument('outputfile')
parser.add_argument('explanation_files', nargs='+')
parser.add_argument('--whitelist', default=None)
parser.add_argument('--blacklist', default=None)
parser.add_argument('--keep_incorrect', action='store_true')
parser.add_argument('--l2_normalize', action='store_true')
parser.add_argument('--max_per_doc', type=int, default=None)
parser.add_argument('--selection_threshold', type=float, default=0.6)
parser.add_argument('--items_in_list', type=int, default=1000)
args = parser.parse_args()


pprint.pprint(vars(args))

def normalize(s):
        s = s.strip(string.punctuation)
        s = s.lower()
        return s


def handle_instance(instance: dict):
    if 'error' in instance:
        return []
    elif not instance['correct'] and not args.keep_incorrect:
        return []

    attribs = {}
    tokens = map(normalize, instance['tokens'])
    for tok, score in zip(tokens, instance['attribs']):
        attribs[tok] = max(attribs.get(tok, -float('inf')), score)
    if args.l2_normalize:
        norm = sum(score ** 2 for score in attribs.values()) ** .5
        attribs = {tok: score / norm for tok, score in attribs.items()}
    selected = [
        {
            'token': tok,
            'score':score,
            'label': instance['gold_label'],
        } 
        for tok, score in sorted(
            attribs.items(),
            reverse=True,
            key=lambda it: it[-1],
        )[:args.max_per_doc]
    ]
    return selected


def handle_predfile(predfile: str):
    with open(predfile, 'r') as istr:
        instances = map(json.loads, istr)
        records = map(handle_instance, instances)
        records = itertools.chain.from_iterable(records)
        df = pd.DataFrame.from_records(records)
        df['source'] = predfile
    return df



data = map(handle_predfile, tqdm.tqdm(args.explanation_files, desc='parsing preds'))
data = pd.concat(list(data))

print('extracting stable tokens')
freq_data = data.groupby(['token', 'label'])['source'].unique().reset_index()
freq_data['selection_freq'] = freq_data['source'].apply(len) / len(args.explanation_files)
stable_tokens_data = freq_data[freq_data['selection_freq'] >= args.selection_threshold]
stable_tokens = set(stable_tokens_data['token'].to_list())


print('aggregating scores')
# SACX defaults to a mean score per all instances of a word
mean_attribs = data[data['token'].isin(stable_tokens)]
mean_attribs = mean_attribs.groupby(['token', 'label'])['score'].mean().reset_index()
mean_attribs = mean_attribs.merge(stable_tokens_data[['token', 'label', 'selection_freq']])

if args.blacklist:
    with open(args.blacklist, 'r') as istr:
        istr = (line.split('\t')[-1].strip() for line in istr)
        blacklist = set(map(normalize, istr))
    mean_attribs['in_blacklist'] =  mean_attribs['token'].isin(blacklist)

if args.whitelist:
    with open(args.whitelist, 'r') as istr:
        istr = (line.split('\t')[-1].strip() for line in istr)
        whitelist = set(map(normalize, istr))
    if args.blacklist:
        # given normalization
        whitelist -= blacklist
    mean_attribs['in_whitelist'] =  mean_attribs['token'].isin(whitelist)

merged_data = (
    mean_attribs
    .sort_values(['label','score'], ascending=[True, False])
    .groupby('label', as_index=False)
    .head(args.items_in_list)
)

merged_data.to_csv(args.outputfile, index=False)
    
    
