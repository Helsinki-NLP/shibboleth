import itertools
import pickle

import numpy as np
import pandas as pd



"""bcms_setimes_overview = pd.read_csv('bcms_setimes_overview.csv', index=0)
bcms_twitter_overview = pd.read_csv('bcms_twitter_overview.csv', index=0)
finnish_suomi24_overview = pd.read_csv('finnish_suomi24_overview.csv', index=0)
german_jodel_overview = pd.read_csv('german_jodel_overview.csv', index=0)
greek_dialects_overview = pd.read_csv('greek_dialects_overview.csv', index=0)
scandinavian_slide_overview = pd.read_csv('scandinavian_slide_overview.csv', index=0)
estonian_voro_overview = pd.read_csv('estonian_voro_overview.csv', index=0)
"""

def method_dist(counts_dict, method_name):
    return {
        (k, i): v[i] 
        for k,v in counts_dict[method_name].items() for i in range(4)
    }

def tvd(dist_a: dict, dist_b:dict) -> float:
    keys = list(set(dist_a.keys()) | set(dist_b.keys()))
    
    dist_a = np.array([dist_a.get(k, 0) for k in keys], dtype=float)
    dist_a /= dist_a.sum()
    dist_b = np.array([dist_b.get(k, 0) for k in keys], dtype=float)
    dist_b /= dist_b.sum()
    
    return 0.5 * abs(dist_a - dist_b).sum()

groups = ['bcms_twitter', 'finnish_suomi24', 'german_jodel', 'greek_dialects',  'scandinavian_slide', 'estonian_voro', 'bcms_setimes']

records = []

for group in groups:
    data_df = pd.read_csv(f'tables/{group}_perclass.csv').rename(columns={'method': 'cfg', 'label': 'lbl', 'Lexical shibboleths': 'Lexical', 'Non-lexical shibboleths': 'Non-lexical', 'Non-shibboleths': 'Non shibboleth'})
    counts = {cfg: {} for cfg in data_df.cfg.unique()}
    for record in data_df.to_dict(orient='records'):
        counts[record['cfg']][record['lbl']] = [record['Lexical'],record['Non-lexical'],record['Non shibboleth'],record['Other']]

# data_cs = pd.read_csv('cs_setimes.csv').rename(columns={'Unnamed: 0': 'cfg', 'Unnamed: 1': 'lbl'})
# counts_cs = {cfg: {} for cfg in data_cs.cfg.unique()}
# for record in data_cs.to_dict(orient='records'):
#     counts_cs[record['cfg']][record['lbl']] = [record['Lexical'],record['Non-lexical'],record['Non shibboleth'],record['Other']]
# with open('cs_setimes.pkl', 'wb') as f:
#     pickle.dump(counts_cs, f)

# for group in groups:
#     with open(f'{group}.pkl', 'rb') as f:
#         counts = pickle.load(f)

    records += [
        {
            'a': a,
            'b': b,
            'tvd': tvd(method_dist(counts, a), method_dist(counts, b)),
            'same_p': a.split('_')[1] == b.split('_')[1],
            'same_xai': a.split('_')[0] == b.split('_')[0],
            'group': group,
        }
        for a, b 
        in itertools.combinations(counts, 2)
    ]

df = pd.DataFrame.from_records(records)
df.to_csv('tables/tvd_results.csv')
