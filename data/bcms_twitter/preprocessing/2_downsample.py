import csv, random, re, sys

infile = sys.argv[1]
outfile = sys.argv[2]

bs = []
me = []
hr = []
sr = []
output = []

with open(infile, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    for row in reader:
        id, lang, instance = row
        if lang == 'bs':
            bs.append(row)
        elif lang == 'me':
            me.append(row)
        elif lang == 'hr':
            hr.append(row)
        elif lang == 'sr':
            sr.append(row)
        else:
            output.append(row) # keeping all multilabel instances


## Downsampling Serbian to the size of the Croatian class
n = len(hr)
sr_ds = random.sample(sr, n)

output.extend(me)
output.extend(bs)
output.extend(hr)
output.extend(sr_ds)
random.shuffle(output)


with open(outfile, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')
    for row in output:
        writer.writerow(row)