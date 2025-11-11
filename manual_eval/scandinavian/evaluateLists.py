import pandas as pd

LIST_PATH="../../data/scandinavian/"

def x_to_bool(s):
	return s == "x"

annotations = pd.read_csv("scandinavian_unique_annotated.csv",
						  converters={"not a shib": x_to_bool, "not minimal": x_to_bool, "regular morph": x_to_bool, "regular phon/spell": x_to_bool, "NE": x_to_bool, "typo/tok/unk": x_to_bool, "lexical": x_to_bool, "counterex": x_to_bool})
annotations = annotations.drop(columns=["Unnamed: 4", "Unnamed: 9", "mark", "count"])
annotations = annotations.rename(columns={"not a shib": "NotShib", "not minimal": "NotMinimal", "regular morph": "Morph", "regular phon/spell": "Phon_Spell", "typo/tok/unk": "Typo_Unk", "lexical": "LexShib", "counterex": "Counterex"})
annotations = annotations.fillna("")
annotations["AnyShib"] = annotations["LexShib"] | annotations["NotMinimal"] | annotations["Morph"] | annotations["Phon_Spell"]

with open(LIST_PATH + "filtered_blacklist.txt") as bl:
	blacklist = [x.strip() for x in bl.readlines()]
	bl_df = pd.DataFrame(data=blacklist, columns=["token"])
	bl_df["in_black"] = True

	bl_join = pd.merge(bl_df, annotations, how='right', on=["token"], suffixes=(None, "_y"))
	bl_join = bl_join.drop(columns=["cognate_da", "cognate_nb", "cognate_nn", "cognate_sv", "NotMinimal", "Morph", "Phon_Spell", "NE", "Typo_Unk", "Counterex"])
	bl_join = bl_join.fillna(False)

	c_annotated_as_notshib = (bl_join['NotShib']).sum()
	c_in_blacklist = (bl_join['in_black']).sum()
	c_true_positive = (bl_join['in_black'] & bl_join['NotShib']).sum()
	# Precision and recall of the blacklist with respect to the manual evaluation
	prec = c_true_positive / c_annotated_as_notshib
	rec = c_true_positive / c_in_blacklist
	f1 = 2 * prec * rec / (prec + rec)
	print("BLACKLIST")
	print(f"Precision: {100*prec:.2f}%")
	print(f"Recall:    {100*rec:.2f}%")
	print(f"F1-score:  {100*f1:.2f}%")

with open(LIST_PATH + "filtered_whitelist.txt") as wl:
	whitelist = [tuple(x.strip().split("\t")) for x in wl.readlines()]
	wl_df = pd.DataFrame(data=whitelist, columns=["label", "token"])
	wl_df["in_white"] = True
	
	wl_join = pd.merge(wl_df, annotations, how='right', on=["token", "label"], suffixes=(None, "_y"))
	wl_join = wl_join.drop(columns=["cognate_da", "cognate_nb", "cognate_nn", "cognate_sv", "NotMinimal", "Morph", "Phon_Spell", "NE", "Typo_Unk", "Counterex"])
	wl_join = wl_join.fillna(False)

	c_annotated_as_shib = (wl_join['AnyShib']).sum()
	c_in_whitelist = (wl_join['in_white']).sum()
	c_true_positive = (wl_join['in_white'] & wl_join['AnyShib']).sum()
	# Precision and recall of the whitelist with respect to the manual evaluation
	prec = c_true_positive / c_annotated_as_shib
	rec = c_true_positive / c_in_whitelist
	f1 = 2 * prec * rec / (prec + rec)
	print("WHITELIST")
	print(f"Precision: {100*prec:.2f}%")
	print(f"Recall:    {100*rec:.2f}%")
	print(f"F1-score:  {100*f1:.2f}%")
