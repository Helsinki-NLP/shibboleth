import random, unicodedata

#    71308 cretan.txt
#   113377 cypriot.txt
#     2716 nothern_version_a.txt
#      842 nothern_version_b.txt
#    67331 pontic.txt

# set max to second-most frequent variant
MAX_LINES = 71308
random.seed(MAX_LINES)	# why not use this number as a seed ;)

def proportion_of_greek_script(text):
	scripts = []
	for x in text:
		if x != " ":
			try:
				n = unicodedata.name(x)
			except:
				n = "UNKNOWN"
			if " " in n:
				scripts.append(n.split(" ")[0])
			else:
				scripts.append(n)
	#scripts = [unicodedata.name(x).split(" ")[0] for x in text if x != " "]
	prop = 0 if scripts.count("GREEK") == 0 else scripts.count("GREEK") / len(scripts)
	return prop

variants = {"CRE": "cretan.txt", "CYP": "cypriot.txt", "NOR": "nothern_version_a.txt", "PON": "pontic.txt"}

all_data = []
for variant, filename in variants.items():
	with open(filename, 'r') as infile:
		data = []
		for line in infile:
			if proportion_of_greek_script(line.strip()) < 0.75:
				print("Skip:", line.strip())
				continue
			data.append(line.strip())
		random.shuffle(data)
		if len(data) > MAX_LINES:
			data = data[:MAX_LINES]
		all_data.extend([(x, variant) for x in data])
random.shuffle(all_data)

with open("grdc_all.csv", "w") as outfile:
	for item in all_data:
		outfile.write(f"{item[1]}\t{item[0]}\t\n")

print("Instances:", len(all_data))
