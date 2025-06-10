import sacremoses, json, re

tokenizers = {
	"da": sacremoses.MosesTokenizer(lang='da'),
	"sv": sacremoses.MosesTokenizer(lang='sv'),
	"nb": sacremoses.MosesTokenizer(lang='nb'),
	"nn": sacremoses.MosesTokenizer(lang='nb'),
}
mpn = sacremoses.MosesPunctNormalizer()
url_regex = re.compile(r'(https?):\/\/(www\.)?\w+[\w/\-?=%.]+\.[\w/\-&?=%.]+')
url2_regex = re.compile(r'www\.\w+[\w/\-?=%.]+\.[\w/\-&?=%.]+')
email_regex = re.compile(r'(\w|\.)+@(\w|\.)+\.(\w\w\w?)')


def normalize(s):
	s = mpn.normalize(s.strip())
	s = s.replace("\t", " ")
	s = url_regex.sub('URL', s)
	s = url2_regex.sub('URL', s)
	s = email_regex.sub('EMAIL', s)
	return s

def tokenize(s, lg):
	toklist = tokenizers[lg].tokenize(s)
	ntok = len(toklist)
	tokstr = " ".join(toklist)
	tokstr = tokstr.replace("&quot;", '"').replace("&apos;", "'").replace("&#91;", "[").replace("&#93;", "]").replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
	tokstr2, nsub = re.subn(r"(\w) '", r"\1'", tokstr)
	return ntok-nsub, tokstr2


def extractJsonl(filename, keep_multilabel):
	tok_counter = {"da": 0, "sv": 0, "nb": 0, "nn": 0, "all": 0}
	sent_counter = {"da": 0, "sv": 0, "nb": 0, "nn": 0, "all": 0}
	outdata = []
	with open(filename) as f:
		for line in f:
			data = json.loads(line)
			if type(data["original"]) is list:
				data["original"] = data["original"][0]
			if data["languages"] == [] or data["original"] == "other" or "other" in data["languages"]:
				continue
			if (not keep_multilabel) and len(data["languages"]) > 1:
				continue
			
			text = normalize(data["text"])
			ntok, toktext = tokenize(text, data["original"])
			for l in data["languages"]:
				tok_counter[l] += ntok
				sent_counter[l] += 1
			tok_counter["all"] += ntok
			sent_counter["all"] += 1
			outdata.append((",".join(data["languages"]), text, toktext))
	return tok_counter, sent_counter, outdata


def extractSlide():
	toks1, sents1, data1 = extractJsonl("slide/training_data/ud/multilabel_ud_sentences_v2.jsonl", keep_multilabel=False)
	toks2, sents2, data2 = extractJsonl("slide/training_data/tatoeba/multilabel_tatoeba_sentences_v2.jsonl", keep_multilabel=False)
	toks3, sents3, data3 = extractJsonl("slide/validation_data/validation_annotated.jsonl", keep_multilabel=False)
	toks4, sents4, data4 = extractJsonl("slide/test_data/test_other_2_new.jsonl", keep_multilabel=False)
	for lang in toks1.keys():
		print(lang, sents1[lang] + sents2[lang] + sents3[lang] + sents4[lang], toks1[lang] + toks2[lang] + toks3[lang] + toks4[lang])
	
	sentences2labels = {}
	for item in data1+data2+data3+data4:
		if item[1] in sentences2labels:
			print("Duplicate")
			sentences2labels[item[1]].append((item[0], item[2]))
		else:
			sentences2labels[item[1]] = [(item[0], item[2])]
	
	with open(f"slide_sl_all_tok.csv", "w") as f_out:
		for sentence, metadata in sentences2labels.items():
			if len(metadata) != 1:
				languages = [x[0] for x in metadata]
				if len(set(languages)) == 1:	# duplicates of the same language => keep one
					f_out.write(f"{metadata[0][0]}\t{sentence}\t{metadata[0][1]}\n")
				else:							# duplicates of different languages => skip
					print("Skip:", sentence, metadata)
					continue
			else:
				f_out.write(f"{metadata[0][0]}\t{sentence}\t{metadata[0][1]}\n")


if __name__ == "__main__":
	extractSlide()
