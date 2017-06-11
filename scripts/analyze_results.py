import pandas as pd
import numpy as np
import csv

results = pd.read_csv('results/run0/test_high_qual_images.csv', sep='\t')
results['prediction'] = results['prediction'].apply(lambda x: x.replace("\'","").replace("[","").replace("]","").split(","))
results['prediction1_genus'] = results['prediction'].apply(lambda x: x[0][0:x[0].find(" ")].title())
results['prediction1_score'] = results['prediction'].apply(lambda x: float(x[0][x[0].find("0"):x[0].find(")")]))

results['prediction2_genus'] = results['prediction'].apply(lambda x: x[1][1:x[1].strip(" ").find(" ") + 1].title())
results['prediction2_score'] = results['prediction'].apply(lambda x: float(x[1][x[1].find("0"):x[1].find(")")]))

results['prediction3_genus'] = results['prediction'].apply(lambda x: x[2][1:x[2].strip(" ").find(" ") + 1].title())
results['prediction3_score'] = results['prediction'].apply(lambda x: float(x[2][x[2].find("0"):x[2].find(")")]))

results['prediction4_genus'] = results['prediction'].apply(lambda x: x[3][1:x[3].strip(" ").find(" ") + 1].title())
results['prediction4_score'] = results['prediction'].apply(lambda x: float(x[3][x[3].find("0"):x[3].find(")")]))

results['prediction5_genus'] = results['prediction'].apply(lambda x: x[4][1:x[4].strip(" ").find(" ") + 1].title())
results['prediction5_score'] = results['prediction'].apply(lambda x: float(x[4][x[4].find("0"):x[4].find(")")]))

results['top1_match'] = results.apply(top_1, axis=1)
results['top5_match'] = results.apply(top_5, axis=1)

top1 = results.groupby(['genus'])['top1_match'].sum()
top1.sort()
top5 = results.groupby(['genus'])['top5_match'].sum()
top5.sort()

accuracy = pd.concat([top1, top5], axis=1)

def top_1(row):
	if row['prediction1_genus'] == row['genus']:
		return 1
	else:
		return 0

def top_5(row):
	if row['genus'] in [row['prediction1_genus'], row['prediction2_genus'], row['prediction3_genus'], row['prediction4_genus'], row['prediction5_genus']]:
		return 1
	else:
		return 0