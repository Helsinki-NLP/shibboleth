# import argparse
# import collections
# import itertools
# import pathlib
import string
import pprint

# import pandas as pd
# import tqdm
import numpy as np


def normalize(s):
    s = s.strip(string.punctuation)
    s = s.lower()
    return s


def lens_from_corpus(infile):
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
    return np.array(list(map(len, dedup.values())))

DATADIR = '../../data'

fnames = {
    'estonian_voro': DATADIR+'/estonian_voro/pkev_all_tok.csv',
    'scandinavian': DATADIR+'/scandinavian/slide_sl_all_tok.csv',
    'BCMS setimes': DATADIR+'/cs_setimes/all_tok.tsv',
    'BCMS twitter': DATADIR+'/bcms_twitter/all_tok.tsv',
    'finnish': DATADIR+'/finnish/suomi24_all_tok.csv',
    'greek': DATADIR+'/greek/grdc_all.csv',
    'german': DATADIR+'/jodel/jodel_all_tok.csv',
    'MLS': DATADIR+'/middle_low_saxon/ReN_northsaxon_all.txt'
}

lens = {k: lens_from_corpus(v) for k, v in fnames.items()}
pprint.pprint({k: np.quantile(v, 1/4) for k,v in lens.items()})
