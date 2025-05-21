#!/usr/bin/env python
# coding: utf-8

import random, sys

# INPUT: all existing splits of a given corpus
# OUTPUT: k folds of the totality of the data
# no changes to file contents
# output file names: [dev|test|train]_fold[1-k].csv
# USAGE: $ python3 make_cross_valid_data.py [k (number of folds)] [file1 file2...]

#k = 10
k = int(sys.argv[1])
#infiles = ['setp_dev.csv', 'setp_test.csv', 'setp_train.csv']
infiles = sys.argv[2:]
data = {}
counter = 0
#seen = []
for infile in infiles:
    print("Processing ", infile)
    with open(infile, 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]        

    for line in lines:
        label, raw, tok = line.split('\t')
        try:
            data[label].append(line)
        except KeyError:
            data[label] = [line]

# remove duplicates within classes
# NB: an instance is considered as duplicate
# only if it carries the same label in both occurrences
# because there can be legitimately ambiguous instances
# between different varieties

for label in data:
    ori_size = len(data[label])
    data[label] = set(data[label])
    data[label] = list(data[label])
    new_size = len(data[label])
    counter  += ori_size - new_size

if counter != 0:
    print("There were duplicates in the data. Instances removed:" , counter)

def get_split_ranges(sizes):
    ranges = []
    i = 0
    for s in sizes:
        j = i + s
        ranges.append((i, j))
        i = j
    return ranges

# Shuffle data in each class
for label in data:
    random.shuffle(data[label])

# Determine data slices
ranges = {}
for label in data:
    print('Processing class:', label)
    total = len(data[label])
    basic_split = total // k
    remainder = total % k
    print('Basis for split size:', basic_split)
    print()
    #print(basic_split, remainder)
    split_sizes = []
    
    for i in range(0, k):
        if remainder > 0:
            split_sizes.append(basic_split + 1)
            remainder -= 1
        else:
            split_sizes.append(basic_split)
            
    ranges[label] = get_split_ranges(split_sizes)  

print('Data slices for each class')
print(ranges)

# Determine slice attribution for each fold
for i in range(k):
    dev = i
    if dev == k - 1:
        test = 0
    else:
        test = i + 1
    train = [n for n in range(k) if (n != dev and n != test)]
    print("=======\nFold " + str(i+1))
    print("Slice attribution")
    print("dev", dev, sep = '\t')
    print("test", test, sep = '\t')
    print("train", train, sep = '\t')
    

    dev_out = []
    test_out = []
    train_out = []
    for label in ranges:
        dev_s, dev_e = ranges[label][dev]
        for instance in data[label][dev_s : dev_e].copy():
            dev_out.append(instance)


        test_s, test_e = ranges[label][test]
        for instance in data[label][test_s : test_e].copy():
            test_out.append(instance)
        
        for r in train:
            #print(label, r)
            train_s, train_e = ranges[label][r]
            for instance in data[label][train_s : train_e].copy():
                train_out.append(instance)
                
    random.shuffle(dev_out)
    random.shuffle(test_out)
    random.shuffle(train_out)

    print("Split sizes")
    print("dev", "test", "train", sep = '\t')
    print(len(dev_out), len(test_out), len(train_out), sep = '\t')
    print()
    
    with open('dev_fold' + str(i + 1) + '.csv', 'w') as f:
        for line in dev_out:
            f.write(line + '\n')

    with open('test_fold' + str(i + 1) + '.csv', 'w') as f:
        for line in test_out:
            f.write(line + '\n')

    with open('train_fold' + str(i + 1) + '.csv', 'w') as f:
        for line in train_out:
            f.write(line + '\n')

print("Done")