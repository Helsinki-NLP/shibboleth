import itertools
import pathlib
import json
import pandas as pd
import tqdm

groups = ['bcms_setimes', 'scandinavian', 'estonian_voro']
methods = ['ig', 'shap', 'loo', 'lime']

def worst_in_instance(row):
    worst, *_ = sorted(zip(row['tokens'], row['attribs'], range(len(row['tokens']))), key=lambda t: t[1])
    tok, attrib, idx = worst
    return {'tok': tok, 'attrib': attrib, 'idx': idx}

base_dir = pathlib.Path('shibboleth/models')
for group in tqdm.tqdm(groups, desc='groups'):
    for method in tqdm.tqdm(methods, desc='XAI', leave=False):
        data = []
        for filename in tqdm.tqdm(list((base_dir / group).glob(f'**/{method}_token.jsonl')), desc='files', leave=False):
            df = pd.read_json(filename, lines=True)
            df['filename'] = str(filename)
            worst = df.apply(worst_in_instance, axis=1)
            for col in ['tok', 'attrib', 'idx']:
                df[f'worst_{col}'] = worst.apply(lambda d: d[col]) 
            data.append(df[df.correct].sort_values('worst_attrib').groupby('gold_label').head(10))
        pd.concat(data).to_json(f'worst_item.{group}.{method}.jsonl', lines=True, orient='records')

del data

for group in tqdm.tqdm(groups, desc='groups'):
    for method in tqdm.tqdm(methods, desc='XAI', leave=False):
        records = []
        for filename in tqdm.tqdm(list((base_dir / group).glob(f'**/{method}_token.jsonl')), desc='files', leave=False):
            with open(filename, 'r') as istr:
                for instance in map(json.loads, istr):
                    baserecord = {k: instance[k] for k in ['correct', 'gold_label']}
                    # baserecord['filename'] = str(filename)
                    for attrib, tok in zip(instance['attribs'], instance['tokens']):
                        records.append({
                            'token': tok,
                            'attrib': attrib, 
                            **baserecord,
                        })
        pd.DataFrame.from_records(records).to_csv(f'inspect.all_attribs.{group}.{method}.csv', index=False)

