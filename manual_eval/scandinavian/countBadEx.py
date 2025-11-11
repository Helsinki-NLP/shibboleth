import pandas as pd

data = {}
for i in range(1, 9):
	listdf = pd.read_csv(f"scandinavian.annotated.{i}.csv")
	count = (listdf.Counterex).sum()
	data[i] = [count]

df = pd.DataFrame.from_dict(data, orient='index', columns=["Count"])
print(df)

#    Count
# 1      1
# 2      0
# 3      0
# 4      0
# 5      1
# 6      1
# 7      1
# 8      3
