import pandas as pd
import os

LIST_PATH="../../data/estonian_voro/"

def x_to_bool(s):
	return s == "x"

# get rid of warning
pd.set_option('future.no_silent_downcasting', True)

annotations_all = None
for listname in [x for x in os.listdir(".") if x.startswith("estonian_voro.list")]:
	annotations = pd.read_csv(listname, sep="\t", header=0)
	if annotations_all is None:
		annotations_all = annotations.copy()
	else:
		annotations_all = pd.concat([annotations_all, annotations]).drop_duplicates().reset_index(drop=True)

annotations_all["Unk"] = False
annotations_all.loc[(annotations_all['Unknown'].str.contains('x', na=False) | annotations_all['Other'].str.contains('x', na=False)), "Unk"] = True
annotations_all["NotShib"] = False
annotations_all.loc[((annotations_all["Unk"] == False) & annotations_all['Unmarked'].str.contains('x', na=False)), "NotShib"] = True
annotations_all["AnyShib"] = False
annotations_all.loc[((annotations_all["Unk"] == False) & (annotations_all["NotShib"] == False) & (annotations_all['Lexicon'].str.contains('x', na=False) | annotations_all['Phon-Morph-infl-Morph-der-Spelling'].str.contains(r'x|s', na=False))), "AnyShib"] = True
annotations_all = annotations_all.drop(columns=["base form", "cognate", "Phon-Morph-infl-Morph-der-Spelling", "Lexicon", "NE", "Other", "Unmarked", "Unknown", "Comment"])

bl_df = pd.read_csv(LIST_PATH + "filtered_blacklist.txt", sep="\t", names=["token"])
bl_df["in_black"] = True
df_join1 = pd.merge(bl_df, annotations_all, how='right', on=["token"], suffixes=(None, "_b"))

wl_df = pd.read_csv(LIST_PATH + "filtered_whitelist.txt", sep="\t", names=["label", "token"])
wl_df["in_white"] = True
df_join = pd.merge(wl_df, df_join1, how='right', on=["token", "label"], suffixes=(None, "_w"))
df_join = df_join.fillna(False)

results = {}
for label in ("est", "vro"):
	df_label = df_join[df_join["label"] == label]
	results[label] = {
		"annotated_items": df_label.shape[0],
		"blacklisted": (df_label['in_black']).sum(),
		"whitelisted": (df_label['in_white']).sum(),
		"non_shibboleths": (df_label['NotShib']).sum(),
		"shibboleths": (df_label['AnyShib']).sum(),
		"blacklisted_shib": (df_label['in_black'] & df_label['AnyShib']).sum(),
		"whitelisted_nonshib": (df_label['in_white'] & df_label['NotShib']).sum()
	}

results["all"] = {
	"annotated_items": df_join.shape[0],
	"blacklisted": (df_join['in_black']).sum(),
	"whitelisted": (df_join['in_white']).sum(),
	"non_shibboleths": (df_join['NotShib']).sum(),
	"shibboleths": (df_join['AnyShib']).sum(),
	"blacklisted_shib": (df_join['in_black'] & df_join['AnyShib']).sum(),
	"whitelisted_nonshib": (df_join['in_white'] & df_join['NotShib']).sum()
}

results_df = pd.DataFrame.from_dict(results)
print(results_df)
print()

errors_df = pd.DataFrame()
errors_df["blacklist error rate"] = results_df.loc["blacklisted_shib"] / results_df.loc["blacklisted"]
errors_df["whitelist error rate"] = results_df.loc["whitelisted_nonshib"] / results_df.loc["whitelisted"]
print(errors_df.T)
