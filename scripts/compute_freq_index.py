import collections
import itertools
import json
import pandas as pd

FILES = [
    ('scandinavian', 'shibboleth/data/scandinavian/slide_sl_all_tok.csv'),
    ('estonian_voro', 'shibboleth/data/estonian_voro/pkev_all_tok.csv'),
    ('bcms_setimes', 'explainability/data_groundtruth/bcms/setimes_large/all_tok.csv'),
]

for grp, file in FILES:
    data = pd.read_csv(file, header=None, names=['label', 'raw', 'tok'], sep='\t')
    for label, subdf in data.groupby('label'):
        freqs = dict(collections.Counter(itertools.chain.from_iterable(subdf.dropna().tok.str.split().to_list())))
        with open(f'freq.{grp}.{label}.json', 'w') as ostr:
            json.dump(freqs, ostr, ensure_ascii=False)
