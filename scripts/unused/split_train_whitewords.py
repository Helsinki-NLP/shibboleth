#!/usr/bin/env python
# coding: utf-8

import sys, json, collections, random
import matplotlib.pyplot as plt
from statistics import mean

#train_file = 'train_tok.csv'
#feat_file = 'train_features.jsonl'
train_file = sys.argv[1]
feat_file = sys.argv[2]

counted = []
with open(train_file, 'r') as tf:
    with open(feat_file, 'r') as ff:
        for trainrow, featrow in zip(tf, ff):
            label, raw, tok = trainrow.split('\t')
            feats = json.loads(featrow)
            white_count = len(feats['white'])
            counted.append((label, raw, white_count))

counted_sorted = sorted(counted, key=lambda tup: tup[2])
counted_sorted[100:200]

ww_counts = collections.Counter(c[2] for c in counted)
print('Distribution of white words per instance')
print('No white words', 'Instances', sep = '\t')
for k in ww_counts:
    print(k, ww_counts[k], sep='\t')
print()


# Splitting into bins
#bin_size = 38746 # determined based on the number of instances (nb training instances / 5)
nb_bins = 5
bin_size = int(round(len(counted) / nb_bins, 0))

level1 = counted_sorted[: bin_size]
level2 = counted_sorted[bin_size : bin_size*2]
level3 = counted_sorted[bin_size*2 : bin_size*3]
level4 = counted_sorted[bin_size*3 : bin_size*4]
level5 = counted_sorted[bin_size*4 : ]
rand_bin = random.sample(counted_sorted, k=bin_size)


print('Bin size', 'Avg whiteword per inst', sep = '\t')
for i, bin in enumerate((level1, level2, level3, level4, level5)):
    bin_num = i + 1
    avg_ww = round(mean([x[2] for x in bin]), 2)
    print('bin' + str(bin_num) + ':', len(bin), avg_ww, sep = '\t')

avg_ww = round(mean([x[2] for x in rand_bin]), 2)
print('randbin:', len(rand_bin), avg_ww, sep = '\t')

for suffix, bin in zip(('_bin1', '_bin2', '_bin3', '_bin4', '_bin5', '_randbin'), (level1, level2, level3, level4, level5, rand_bin)):
    outfile = 'train' + suffix + '.csv'
    random.shuffle(bin)
    with open(outfile, 'w') as o:
        for line in bin:
            o.write('\t'.join((line[0], line[1], str(line[2]))) + '\n')

