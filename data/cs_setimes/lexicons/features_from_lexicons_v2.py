#!/usr/bin/env python
# coding: utf-8

import csv, os, re, sys

hr_file = 'hrLex_v1.3'
sr_file = 'srLex_v1.3'

combined_lex = {}
# combined lexicon collapses ambiguous wordforms
# since in the corpus the only info available is the token,
# frequencies for all entries for a given wordform are combined
# independently of pos, lemma and msd
# by summing up the frequencies

with open(hr_file, newline='') as hr_csv:
    hr_reader = csv.DictReader(hr_csv, fieldnames=['token', 'lemma', 'pos', 'msd', 'upos', 'traits', 'freq', 'relfreq'], delimiter = '\t')
    for row in hr_reader:
            
        if row['freq'] is None or row['relfreq'] is None:
                #print(row)
                continue
        try:
            
            combined_lex[row['token']]['freq_hr'] += float(row['freq'])
            combined_lex[row['token']]['relfreq_hr'] += float(row['relfreq'])
        except KeyError:
            combined_lex[row['token']] = {'freq_hr' : float(row['freq']), 'relfreq_hr' : float(row['relfreq']), 'freq_sr' : 0, 'relfreq_sr' : 0} 

with open(sr_file, newline='') as sr_csv:
    sr_reader = csv.DictReader(sr_csv, fieldnames=['token', 'lemma', 'pos', 'msd', 'upos', 'traits', 'freq', 'relfreq'], delimiter = '\t')
    for row in sr_reader:
        if row['freq'] is None or row['relfreq'] is None:
            #print(row)
            continue

        try:
            combined_lex[row['token']]['freq_sr'] += float(row['freq'])
            combined_lex[row['token']]['relfreq_sr'] += float(row['relfreq'])
        except KeyError:
            combined_lex[row['token']] = {'freq_sr' : float(row['freq']), 'relfreq_sr' : float(row['relfreq']), 'freq_hr' : 0, 'relfreq_hr' : 0}

sr_white = []
hr_white = []
black = []


for lex in combined_lex: ## NB: no frequency thresholding
    if combined_lex[lex]['freq_hr'] > 0 and combined_lex[lex]['freq_sr'] == 0:
        hr_white.append(lex)
    elif combined_lex[lex]['freq_hr'] == 0 and combined_lex[lex]['freq_sr'] > 0:
        sr_white.append(lex)
    elif combined_lex[lex]['freq_hr'] > 0 and combined_lex[lex]['freq_sr'] >0:
        black.append(lex)
    else:
        continue

sr_white.sort()
hr_white.sort()
black.sort()

with open('whitelist.txt', 'w') as f:
    [f.write('sr' + '\t' + w + '\n') for w in sr_white]
    [f.write('hr' + '\t' + w + '\n') for w in hr_white]

with open('blacklist.txt', 'w') as f:
    [f.write(w + '\n') for w in black]

