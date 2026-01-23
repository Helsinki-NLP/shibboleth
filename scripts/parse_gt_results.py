import pathlib
import itertools
import collections

import pandas as pd

records = []
topdir = pathlib.Path('models')
files = itertools.chain(
    pathlib.Path('models').glob('**/ig_eval.txt'),
    pathlib.Path('models').glob('**/shap_eval.txt'),
    pathlib.Path('models').glob('**/loo_eval.txt'),
)
files = pathlib.Path('.').glob('wl_*_eval_gt_instances.txt')
files = pathlib.Path('shibboleth/models').glob('**/*_results+wl.txt')
files = list(files)
print(files)

for file in files:
    with open(file, 'r') as istr:
        data = istr.read().split('\n\n')
        data = (record for record in data if record.strip() != '')
        data = (
            dict(line.split(':') for line in record.strip().split('\n')) 
            for record in data
        )
        data = ({k.strip(): v.strip() for k, v in record.items()} for record in data)
        records_ = list(data)
        for record in records_:
            record['filename'] = str(file)
        records += records_

df = pd.DataFrame.from_records(records)
for perc_col in ['Accuracy', 'Precision', 'Recall', 'F1-score']:
    df[perc_col] = df[perc_col].str.strip('%')
U_tests = df['U-test'].apply(lambda u_test: dict(info.split('=') for info in u_test.split()))
for key in 'Upf':
    df[f'U-test {key}'] = U_tests.apply(lambda dct: dct[key])
del df['U-test']
for float_col in ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUROC'] + [f'U-test {key}' for key in 'Upf']:
    df[float_col] = df[float_col].apply(float)
df['Evaluation instances'] = df['Evaluation instances'].apply(int)

df['lang group'] =  df['Ground truth'].apply(lambda gt: gt.split('/')[5])
df['attrib'] =  df['Predictions'].apply(lambda prd: prd.split('/')[-1])

df.to_csv('auroc_data+wl.csv', index=False)

## data for display
df_ = df[~df['Predictions'].str.contains('tok/')][['lang group', 'attrib', 'Accuracy', 'F1-score', 'Precision', 'Recall', 'AUROC', 'U-test p']].rename(columns={'AUROC': 'U-test f'})

df_ = df_[~df_.attrib.str.startswith('loo2')] # I'm being lazy for now

#attribs_dict = collections.defaultdict(lambda: 'LOO (max)')
#attribs_dict.update({'ig_token.jsonl': 'IG (sum)', 'ig_corr_agg_max.json': 'IG (max)', 'shap_token.jsonl': 'SHAP (sum)', 'shap_corr_agg_max.json': 'SHAP (max)', 'loo_token.jsonl': 'LOO (sum)', 'lime_token.jsonl': 'LIME (sum)'})
attribs_dict = {
    'ig_token.jsonl': 'IG (sum)',
    'ig_corr_agg_max.json': 'IG (max)',
    'shap_token.jsonl': 'SHAP (sum)',
    'shap_corr_agg_max.json': 'SHAP (max)',
    'loo_token.jsonl': 'LOO (sum)',
    'lime_token.jsonl': 'LIME (sum)',
}
df_['attrib'] = df_['attrib'].map(attribs_dict)
print(df_.sort_values(by=['lang group', 'attrib']).to_latex(index=False))
