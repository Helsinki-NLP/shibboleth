#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import html
import csv
import classla
from random import shuffle

df_hbs_setimes = pd.read_json('SETimes.HBS.json')
df_hbs_setimes = df_hbs_setimes[df_hbs_setimes.language != 'bs']
df_hbs_setimes['text'] = df_hbs_setimes['text'].str.strip()

nlp_sr = classla.Pipeline('sr', processors='tokenize')
nlp_hr = classla.Pipeline('hr', processors='tokenize')

hbs_setimes = df_hbs_setimes.to_csv(sep='\t', header=False, index=False)
lines = hbs_setimes.split('\n')

train = []
test = []
dev = []

i = 0
counters = { 'train' : {'sr' :  0, 'hr' : 0}, \
           'test' : {'sr' : 0, 'hr' : 0}, \
            'dev' : {'sr' : 0, 'hr' : 0} }

check = [] # for sanity check
            
for line in lines: 
    i += 1
    try:
        text, lang, split = line.split('\t')
    except:
        print(i, line)
        
    if lang == 'sr':
        conll = nlp_sr(text).to_conll().split('\n')
    elif lang == 'hr':
        conll = nlp_hr(text).to_conll().split('\n')
    else:
        sys.err('Wrong language in dataframe: ' + lang)
        
    for c in conll:
        if c.startswith('# text ='):
            sentence = re.sub(r'^# text = ', '', c)
            check.append(sentence)
            tokens = []
        elif re.match('^\d+\t', c):
            _, token, *_ = c.split('\t')
            tokens.append(token)
            counters[split][lang] += 1
        elif c == '':
            if split == 'train':
                train.append((lang, sentence, ' '.join(tokens)))
            elif split == 'dev':
                dev.append((lang, sentence, ' '.join(tokens)))
            elif split == 'test':
                test.append((lang, sentence, ' '.join(tokens)))
            else:
                print('Error in labels:', split)
        else:
            continue

print(len(test), len(dev), len(train), len(check))

with open('check_sent.txt', 'w') as f:
    [f.write(line + '\n') for line in check]

shuffle(test)
shuffle(train)
shuffle(dev)

with open('test_tok.tsv', 'w') as out:
    for line in test:
        out_line = '\t'.join(line)
        out_line = re.sub('""', '"', out_line)
        out.write(out_line + '\n')

with open('train_tok.tsv', 'w') as out:
    for line in train:
        out_line = '\t'.join(line)
        out_line = re.sub('""', '"', out_line)
        out.write(out_line + '\n')


with open('dev_tok.tsv', 'w') as out:
    for line in dev:
        out_line = '\t'.join(line)
        out_line = re.sub('""', '"', out_line)
        out.write(out_line + '\n')

print(counters)



