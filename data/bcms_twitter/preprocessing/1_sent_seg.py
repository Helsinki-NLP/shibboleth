#!/usr/bin/env python
# coding: utf-8

import sys, re, classla, csv
from random import shuffle


input_file = sys.argv[1]
output_file = sys.argv[2]



with open(input_file, 'r') as f:
    lines = [line.rstrip().split('\t') for line in f.readlines()]

nlp = classla.Pipeline('sr', type='nonstandard', processors='tokenize')
i = 0 # attributing IDs to preserve the link between the original instance and the sentences from it
output = []
for label, instance in lines:
    i += 1
    conll = nlp(instance).to_conll().split('\n')
    for line in conll:
        if line.startswith('# text ='):
            sentence = re.sub(r'^# text = ', '', line)
            if re.search(r'[a-zđšžćčA-ZĐŠŽĆČ]', sentence): # keeping only sentences with at least one letter character
                output.append((i, label, sentence))

shuffle(output)

with open(output_file, 'w', newline='') as csvfile:
    outwriter = csv.writer(csvfile, delimiter='\t')
    for o in output:
        outwriter.writerow(o)

