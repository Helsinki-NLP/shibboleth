import argparse
import collections
import json
import string
import math

import pandas as pd
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('aggregation', choices=['sum', 'relrank'])
parser.add_argument('explanations')
parser.add_argument('whitelist')
parser.add_argument('blacklist')
parser.add_argument('outputfile')
parser.add_argument('--keep_incorrect', action='store_true')
args = parser.parse_args()


import pprint
pprint.pprint(vars(args))

def normalize(s):
        s = s.strip(string.punctuation)
        s = s.lower()
        return s

global_word_count = collections.defaultdict(float)
global_doc_freq = collections.defaultdict(float)
records = []

with open(args.explanations, 'r') as istr:
    data = map(json.loads, istr)
    if not args.keep_incorrect:
        data = (p for p in data if p["correct"])
    for pred_dict in data:
        attribs = collections.defaultdict(float)
        tokens =  list(map(normalize, pred_dict["tokens"]))
        term_freq = {k: v / len(tokens) for k, v in collections.Counter(tokens).items()}
        for idx, word in enumerate(tokens):
            attribs[word] += pred_dict["attribs"][idx]
        if args.aggregation == 'relrank':
            n_items = len(attribs)
            attribs = {
                word: idx / n_items
                for idx, word in enumerate(sorted(attribs, key=attribs.__getitem__), start=1)
            }
        for word in attribs:
            global_doc_freq[word] += 1
        for word in tokens:
            global_word_count[word] += 1
        records.append({'tf': term_freq, 'attribs': dict(attribs), 'tokens': tokens})

for record in records:
    record['idf'] = {
        tok: math.log(len(records) / (global_doc_freq[tok] + 1)) + 1
        for tok in record['tf']
    }
    record['tf.idf'] = {
        tok: record['tf'][tok] * record['idf'][tok]
        for tok in record['tf']
    }



all_scores = []

with open(args.whitelist, 'r') as istr:
    istr = (line.split('\t')[-1].strip() for line in istr)
    whitelist = set(map(normalize, istr))

with open(args.blacklist, 'r') as istr:
    istr = (line.split('\t')[-1].strip() for line in istr)
    blacklist = set(map(normalize, istr))

# given normalization
whitelist -= blacklist

for k_restrict in tqdm.tqdm([5, 10, 20, 50, None]):
    global_scores = {
        'tfidf': collections.defaultdict(float),
        'toknorm': collections.defaultdict(float),
        'docnorm': collections.defaultdict(float),
        'no_weights': collections.defaultdict(float),
    }

    for record in tqdm.tqdm(records, position=1, leave=False):
        best_k = sorted(record['attribs'], key=attribs.__getitem__, reverse=True)[:k_restrict]
        for tok in best_k:
            global_scores['tfidf'][tok] += record['attribs'][tok] * record['tf.idf'][tok]
            global_scores['toknorm'][tok] += record['attribs'][tok] / global_word_count[tok]
            global_scores['docnorm'][tok] += record['attribs'][tok] / global_doc_freq[tok]
            global_scores['no_weights'][tok] += record['attribs'][tok]

    dataframe = []
    for method, score_dict in global_scores.items():
        words, score = zip(*score_dict.items())
        score_df = pd.DataFrame({'word': words, 'score': score}).sort_values('score', ascending=False)
        score_df[f'order'] =  range(1, len(score_df) + 1)
        score_df[f'% in white'] = (score_df['word'].isin(whitelist).cumsum() / score_df.order) * 100
        score_df[f'% in black'] = (score_df['word'].isin(blacklist).cumsum() / score_df.order) * 100
        score_df[f'rel ordering'] = score_df['order'] / len(score_df)
        score_df['method'] = method
        dataframe.append(score_df)
    dataframe = pd.concat(dataframe)
    dataframe['restrict'] = "no" if k_restrict is None else ('top'+str(k_restrict))
    all_scores.append(dataframe)

pd.concat(all_scores).to_csv(args.outputfile)
