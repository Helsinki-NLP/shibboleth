#!/usr/bin/env python
# coding: utf-8

# USAGE: python3 split_train_whitewords_strat.py train_tok.csv train_features.jsonl 5
# OUTPUT: files named train_(no_bins)_bin(1-no_bins).csv + train_(no_bins)_randbin.csv
# OUTPUT format: [label][\t][instance][\t][white word count]\n
# format change should not affect usage of training files based on raw input
# OUTPUT: also prints info about bin size, avg no of white words, etc. to STDOUT

import sys, json, collections, random
from statistics import mean

train_file = sys.argv[1]
feat_file = sys.argv[2]
no_bins = int(sys.argv[3])

# Determining size of each bin
def find_bin_sizes(label, class_sorted):
    class_size = len(class_sorted)
    basis = class_size // no_bins
    modulo = class_size % no_bins
    bin_sizes = []
    
    for i in range(0, no_bins):
        bin_sizes.append(basis)
    
    for i in range(0, modulo):
        bin_sizes[i] += 1
        i += 1
    
    print('Bin sizes:')
    [print(b) for b in bin_sizes]
    print()
    
    return bin_sizes


# Counting white words per instance
counted = {}
with open(train_file, 'r') as tf:
    with open(feat_file, 'r') as ff:
        for trainrow, featrow in zip(tf, ff):
            label, raw, tok = trainrow.split('\t')
            feats = json.loads(featrow)
            #print(raw)
            #print(feats)
            white_count = len(feats['white'])
            if  label in counted:
                counted[label].append((label, raw, white_count))
            else:
                counted[label] = [(label, raw, white_count)]

# Creating stratified bins
output = {}
for label in counted:
    output[label] = []
    class_sorted = sorted(counted[label], key=lambda tup: tup[2])
    class_counts = collections.Counter(c[2] for c in class_sorted)
    print(label)    
    print('# whitewords', 'Instances', sep = '\t')
    for c in class_counts:
        print(c, class_counts[c], sep = '\t')
    print()

    bin_sizes = find_bin_sizes(label, class_sorted)

    buffer = []
    for size in bin_sizes:
        for i in range(0, size):
            inst = class_sorted.pop(0)
            buffer.append(inst)
        output[label].append(buffer)
        buffer = []
    
# Creating merged bins
output_cp = output.copy()
final_out = []
print('Average whiteword count per bin')
for label in output_cp:
    print(label)
    for bin in output_cp[label]:
        avg_ww = round(mean([x[2] for x in bin]), 2)
        print(len(bin), avg_ww, sep = '\t')
    print()

    for i in range(0, len(output_cp[label])): 
        cp = output_cp[label][i].copy()
        try:            
            final_out[i].extend(cp)
        except IndexError:
            final_out.append(cp)

# Printing info about final bins
print('Final bins')
print('Size', 'Avg ww count', 'Classes', sep = '\t')
for bin in final_out:
    #print(len(bin))
    class_counts = collections.Counter(c[0] for c in bin)
    avg_ww = round(mean([x[2] for x in bin]), 2)
    #print(avg_ww)
    #print(class_counts)
    print(len(bin), avg_ww, sep = '\t', end = '\t')
    for cl in class_counts:
        print(cl + '=' + str(class_counts[cl]), end = '\t')
    print()
print()


# Outputting stratified bins
for i in range (0, no_bins):
    random.shuffle(final_out[i])
    out_file = 'train_' + str(no_bins) + 'bins' + '_bin' + str(i+1) + '.csv'
    with open(out_file, 'w') as f:
        for line in final_out[i]:            
            f.write('\t'.join((line[0], line[1], str(line[2]))) + '\n')


# Creating the random bin (still stratified)
rand_bin_inst = {}

for cl in output:
    rand_bin_size = len(output[cl][0])
    rand_bin_inst[cl] = random.sample(counted[cl], k=rand_bin_size)
    
rand_bin = []
for cl in rand_bin_inst:
    rand_bin.extend(rand_bin_inst[cl])

# Info about the random bin
print('Random bin')
print('Size', 'Avg ww count', 'Classes')
class_counts = collections.Counter(c[0] for c in rand_bin)
avg_ww = round(mean([x[2] for x in rand_bin]), 2)
#print(avg_ww)
#print(class_counts)
print(len(rand_bin), avg_ww, sep = '\t', end = '\t')
for cl in class_counts:
    print(cl + '=' + str(class_counts[cl]), end = '\t')
print()
print()


# Outputting the random bin
rand_file = 'train_' + str(no_bins) + 'bins_randbin.csv'
with open(rand_file, 'w') as f:
    for line in rand_bin:
        f.write('\t'.join((line[0], line[1], str(line[2]))) + '\n')

