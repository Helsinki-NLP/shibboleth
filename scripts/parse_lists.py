import argparse
import collections
import itertools
import pathlib
import string

import pandas as pd
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('whitelist')
parser.add_argument('blacklist')
parser.add_argument('corpus')
parser.add_argument('outputfile')
parser.add_argument('filedir', type=pathlib.Path)
args = parser.parse_args()


def normalize(s):
    s = s.strip(string.punctuation)
    s = s.lower()
    return s


with open(args.whitelist, 'r') as istr:
    istr = (line.split('\t')[-1].strip() for line in istr)
    whitelist = set(map(normalize, istr))

with open(args.blacklist, 'r') as istr:
    istr = (line.split('\t')[-1].strip() for line in istr)
    blacklist = set(map(normalize, istr))

# given normalization
whitelist -= blacklist

df = pd.read_csv(
        args.corpus,
        sep='\t',
        header=None,
        names=['label', 'raw', 'tok'],
    )

def read_corpus(infile):
    dedup = dict()
    print("Processing ", infile)
    with open(infile, 'r') as f:
        lines = [line.rstrip('\n\r ') for line in f.readlines()]
        for line in lines:
            label, raw, tok = line.split('\t')
            tokens = tok.strip().split(' ') if tok.strip() != "" else raw.strip().split(' ')
            if len(tokens) < 3:
                continue

            if not raw in dedup:
                dedup[raw] = tuple(map(normalize, tokens))
    return list(dedup.values())

corpus = read_corpus(args.corpus)
corpus = dict(collections.Counter(corpus))


records = []
for filename in tqdm.tqdm(list(args.filedir.glob('*.csv'))):
    df = pd.read_csv(filename)
    df['token_freq'] = df['token'].map(corpus)
    records.append(
        dict(
            prop_wl = df['token'].isin(whitelist).mean(),
            prop_bl = df['token'].isin(blacklist).mean(),
            prop_happaxes = (df.token_freq == 1).mean(),
            prop_lt2 = (df.token_freq <= 2).mean(),
            prop_lt3 = (df.token_freq <= 3).mean(),
            prop_lt4 = (df.token_freq <= 4).mean(),
            prop_lt5 = (df.token_freq <= 5).mean(),
            num_extracted = len(df),
            fname = filename,
        )
    )

df_records = pd.DataFrame.from_records(records).to_csv(args.outputfile, index=False)
