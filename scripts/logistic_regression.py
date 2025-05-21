import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

parser = argparse.ArgumentParser()

parser.add_argument('--trainfile', required=True)
parser.add_argument('--testfile', required=True)
parser.add_argument('--devfile', required=False)
parser.add_argument('--iterations', required=False, default=500)
parser.add_argument('--dtype', required=False, default='int8')
args = parser.parse_args()

#print(type(args.dtype))
#print(type(np.dtype(args.dtype)))

corpus_train = open(args.trainfile, 'r').readlines()[1:]
corpus_test = open(args.testfile, 'r').readlines()[1:]

#in case you want to include vocabulary from the dev file:
#corpora = [corpus_train, corpus_dev]
corpora = [corpus_train]

vocabulary = []
test_vocabulary = []

def collect_vocab(corpus, vocablist):
    for ind, row in enumerate(corpus):
        split = row.split('\t')
        if len(split) == 3:
            tokens = split[2].split(' ')
            for token in tokens:
                if token.strip() not in vocablist:
                    vocablist.append(token.strip())
        elif len(split) == 4:
            tokens = split[3].split(' ')
            for token in tokens:
                if token.strip() not in vocablist:
                    vocablist.append(token.strip())
                
collect_vocab(corpus_train, vocabulary)
collect_vocab(corpus_test, test_vocabulary)


print("Train vocabulary length:", len(vocabulary))
print("Test vocabulary length:", len(test_vocabulary))

vocab_dict = {}
for ind, vocab in enumerate(vocabulary):
    vocab_dict[vocab] = ind

def convert_corpus(corpus, vocab_dict, vocabulary):
    vocab_size = len(vocabulary)
    num_rows = len(corpus) - 1
    converted = np.zeros((num_rows, vocab_size), dtype=np.dtype(args.dtype))
    labels = []
    
    for i, row in enumerate(corpus[1:], start=0):
        split = row.split('\t')
        if len(split) == 3:
            labels.append(split[0])
            sentence = split[2].split(' ')
        elif len(split) == 4:
            labels.append(split[1])
            sentence = split[3].split(' ')
        
        for token in sentence:
            token = token.strip()
            if token in vocab_dict:
                converted[i, vocab_dict[token]] += 1
            
                
    return converted, np.array(labels)

test_corpus, test_labels = convert_corpus(corpus_test, vocab_dict, vocabulary)
train_corpus, train_labels = convert_corpus(corpus_train, vocab_dict, vocabulary)

logisticRegr = LogisticRegression(max_iter=int(args.iterations))

logisticRegr.fit(train_corpus, train_labels)

predictions = logisticRegr.predict(test_corpus)

score = logisticRegr.score(test_corpus, test_labels)
print("Accuracy score:", score)

f1score = metrics.f1_score(test_labels, predictions, average='weighted')
print("F1 score:", f1score)

cm = metrics.confusion_matrix(test_labels, predictions)
print("confusion matrix:")
print(cm)

def Sort_Tuple(tuplist):
    tuplist.sort(key = lambda x: x[1], reverse=True)
    return tuplist

classes_and_coefs = {}
classes = logisticRegr.classes_
#print("classes", classes)

coefs = logisticRegr.coef_
#print(len(coefs))
#print(type(coefs))


if len(coefs) == 1:
    coefs_2 = coefs[0]
    coefs_1 = -coefs_2
    coefs = np.stack((coefs_1, coefs_2), axis=0)
#print(coefs)

for ind1, classe in enumerate(coefs):
    print(ind1)
    classes_and_coefs[classes[ind1]] = []
    coef_values = classe
    
    word_coefs = []
    for ind2, vocab in enumerate(vocabulary):
        word_coefs.append((vocab, coef_values[ind2]))
    
    sorted_word_coefs = Sort_Tuple(word_coefs)
    
    filename_all = args.trainfile[:-3] + str(classes[ind1]) + '_train-coefs.txt'
    file_all = open(filename_all, 'w')
    
    filename_test = args.trainfile[:-3] + str(classes[ind1]) + '_test-coefs.txt'
    file_test = open(filename_test, 'w')
    
    
    ind4 = 0
    for ind3, pair in enumerate(sorted_word_coefs):
        if ind3 < 30:
            file_all.write(pair[0] + '\t' +  str('%.20f' % pair[1]) + '\n')
        if pair[0] in test_vocabulary:
            if ind4 < 30:
                file_test.write(pair[0] + '\t' +  str('%.20f' % pair[1]) + '\n')
                ind4 += 1
    

#f-score in addition to accuracy
#top 30 coefs, and one version with only the coef words that also occur in the test set
