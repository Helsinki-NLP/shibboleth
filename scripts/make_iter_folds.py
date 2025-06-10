#!/usr/bin/env python
# coding: utf-8
import os, random, sys

# INPUT: full content of a given corpus (dev, test, train)
# OUTPUT:  n x k folds of the totality of the data
# OUTPUT STRUCTURE: 
# One master directory: iter_folds/
# One directory per iteration (iter[1-n])
# In each iteration directory:
# - dev and train sets for each fold ({dev, train}_iter[1-n]_fold[1-k].tsv)
# There is only one test set per corpus
# NB: deduplication is done to eliminate multilabel instances and bleedover present in original datasets
# USAGE: $ python3 make_iter_folds.py [k (number of folds)] [n (number of iterations)] [file1 file2...]

#k = 10
k = int(sys.argv[1])
#n = 10
n = int(sys.argv[2])
#infiles = ['setp_dev.csv', 'setp_test.csv', 'setp_train.csv']
infiles = sys.argv[3:]

# Data preprocessing
data = {}
dedup = {}
short_counter = 0
seen = []
ori_size = 0

for infile in infiles:
    print("Processing ", infile)
    with open(infile, 'r') as f:
        lines = [line.rstrip('\n\r ') for line in f.readlines()]
        ori_size += len(lines)
        for line in lines:
            label, raw, tok = line.split('\t')
            
            tokens = tok.strip().split(' ') if tok.strip() != "" else raw.strip().split(' ')
            if len(tokens) < 3:
                short_counter += 1
                continue
                
            if not raw in dedup:
                dedup[raw] = (label, tok)
            else:
                seen.append(raw)

print("Original dataset size:", ori_size)
print("Instances with < 3 tokens:", short_counter)

# Removing all instances that were seen twice
# Needed to eliminate multilabel instances
if len(seen) != 0:
    start = len(dedup)
    seen = set(seen)  
    
    for s in seen:
        del dedup[s]

    print("Duplicate types:", len(seen))
    
print("After deduplication:", len(dedup))

# Group data by class
for raw in dedup:
    label, tok = dedup[raw]
    try:
        data[label].append('\t'.join((label, raw, tok)))
    except KeyError:
        data[label] = ['\t'.join((label, raw, tok))]


# Data splitting

def get_split_ranges(sizes):
    ranges = []
    i = 0
    for s in sizes:
        j = i + s
        ranges.append((i, j))
        i = j
    return ranges

master_dir = "iter_folds"
try:
    os.mkdir(master_dir)
except FileExistsError:
    print(f"Using the existing '{master_dir}' directory.")

# Shuffle data in each class
# Get Dtest and Dexp
d_test = {}
d_exp = {}
for label in data:
    random.shuffle(data[label])
    print('Full class', label, ':', len(data[label]))
    test_size = len(data[label]) // 10
    test = data[label][0 : test_size].copy()
    exp = data[label][test_size : ].copy()    
    print('Extracted for test:', len(test))
    print('Extracted for exp:', len(exp))
    d_test[label] = test
    d_exp[label] = exp
    print()


test_out = []
for label in d_test:
    print('In d_test class', label, ':', len(d_test[label]))
    test_out.extend(d_test[label])
random.shuffle(test_out)
print()

print('Length of test_out:', len(test_out))

with open(master_dir + '/test.tsv', 'w') as f:
    [f.write(line + '\n') for line in test_out]

ranges = {}
for label in d_exp:
    print('Processing class:', label)
    total = len(d_exp[label])
    basic_split = total // k
    remainder = total % k
    print('Basis for split size:', basic_split)
    print()
    split_sizes = []
    
    for i in range(0, k):
        if remainder > 0:
            split_sizes.append(basic_split + 1)
            remainder -= 1
        else:
            split_sizes.append(basic_split)
            
    ranges[label] = get_split_ranges(split_sizes)  

print('Data slices for each class')
for r in ranges:
    print(r, ranges[r])


# Iterating k-folds
for j in range(n):
    print("Starting iteration", j + 1)

    dir = master_dir + "/iter" + str(j + 1)
    try:
        os.mkdir(dir)
    except FileExistsError:
        print(f"Using the existing '{dir}' directory. Overwriting existing files.")

# Shuffling the data
    for label in d_exp:
        random.shuffle(d_exp[label])
    
    for i in range(k):
        dev = i
        train = [s for s in range(k) if s != dev]
        print("=======\nFold " + str(i+1))
        print("Slice attribution")
        print("dev", dev, sep = '\t')
        print("train", train, sep = '\t')    
    
        dev_out = []
        train_out = []
        for label in ranges:
            dev_s, dev_e = ranges[label][dev]
            for instance in d_exp[label][dev_s : dev_e].copy():
                dev_out.append(instance)
            
            for r in train:
                train_s, train_e = ranges[label][r]
                for instance in d_exp[label][train_s : train_e].copy():
                    train_out.append(instance)
    
        print("Split sizes")
        print("dev", "train", sep = '\t')
        print(len(dev_out), len(train_out), sep = '\t')
        print()

        random.shuffle(dev_out)
        random.shuffle(train_out)
        
        with open(dir + '/dev' + '_iter' + str(j + 1) + '_fold' + str(i + 1) + '.tsv', 'w') as f:
            for line in dev_out:
                f.write(line + '\n')
    
        with open(dir + '/train' + '_iter' + str(j + 1) + '_fold' + str(i + 1) + '.tsv', 'w') as f:
            for line in train_out:
                f.write(line + '\n')
    
    print("Done with iteration", j + 1)
    print()

