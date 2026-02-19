#!/usr/bin/env python
# coding: utf-8

# # Goals
# 
# - mask URLs and username handles
# - tokenize
# - correct the format
# - filter out instances with less than 3 tokens that are not URLs and handles

import csv, re, sys
import classla


nlp_sr = classla.Pipeline('sr', processors='tokenize')
nlp_hr = classla.Pipeline('hr', processors='tokenize')


#infile = 'data_twitter_dssr_train_sl_clean.tsv'
infile = sys.argv[1]
#outfile = 'testing_train.tsv'
outfile = sys.argv[2]

label_correspondence = {    
    '0' : 'bs',
    '1' : 'hr',
    '2' : 'me',
    '3' : 'sr'
}

output = []


with open(infile, newline='') as f:
    csvreader = csv.reader(f, delimiter='\t' )
    next(csvreader)
    for row in csvreader:
        if len(row) != 4:
            continue
            
        label = label_correspondence[row[1]]
        if not label in ('me', 'bs', 'hr', 'sr'):
            continue
            
        instance = row[3]
        if re.match(r'^\s*$', instance):
            continue
            
        if re.search(r'[\n\r]', instance):
            continue
        
        pattern = r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*"
        instance = re.sub(pattern, '[URL]', instance)
        pattern = r'@[a-zA-Z0-9_]+\b'
        instance = re.sub(pattern, '[HANDLE]', instance)      
        
        if label == 'sr':
            conll = nlp_sr(instance).to_conll().split('\n')
        elif label == 'hr' or label == 'me' or label == 'bs':
            conll = nlp_hr(instance).to_conll().split('\n')
        else:
            sys.err('Wrong language in dataset: ' + lang)
            
        tokens = []
        for c in conll:
            if re.match(r"\d+\t", c):
                token = c.split('\t')[1]
                tokens.append(token)
        if len(tokens) == 0:
            print(row)
            sys.exit("No tokens after tokenization")
            
        tokenized = " ".join(tokens)
        tokenized = re.sub('\[ HANDLE \]', '[HANDLE]', tokenized)
        tokenzed = re.sub(r'\[ URL \]', '[URL]', tokenized)
        
        retok = tokenized.split(' ')             
        valid_tokens = 0
        for t in retok:
            if t != '[HANDLE]' and t != '[URL]':
                valid_tokens += 1
        if valid_tokens < 3:
            continue
            
        output.append((label, instance, tokenized))

with open(outfile, 'w') as f:
    for o in output:
        f.write("\t".join(o) + '\n')

