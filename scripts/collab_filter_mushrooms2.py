# read input CSV from MO

import pandas as pd
import numpy as np
import csv
from scipy.spatial.distance import cosine

# ref: http://www.salemmarafi.com/code/collaborative-filtering-with-python/

# In the traditional user-item scheme of CF
# our users (rows) are observations
# our items (columns) are mushroom species, variations, groups, clades etc.
# v_i,j = 1 if vote_cache for obs i as species j is > threshold. else 0
# for the items, we will exclude genera but include all classifications more specific than genera (i.e. rank 8 and under)
votes = pd.read_csv('full_votes_for_CF.csv', sep='\t')

# get name info from master and name mappings
master = pd.read_csv('../master_image_data.csv', sep='\t')
name_mapping = pd.read_csv('name_to_official_mapping.csv', sep='\t') 

# use official names for all vote proposals
votes_official_names = pd.merge(votes, name_mapping, left_on='name_id', right_on='id_temp')
votes_official_names = votes_official_names.drop(['name_id', 'Unnamed: 0', 'id_temp'], axis=1)
 
# get text-name and rank for each name id
name_info = pd.DataFrame(master.groupby(['official_name_id', 'text_name', 'rank']).groups.keys(), columns = ['name_id', 'text_name', 'rank'])

#acceptable ranks
name_info_under_genus = name_info.loc[(name_info['rank'] < 9) | (name_info['rank'] == 16)]


# get votes with official names < rank genus 
votes_master = pd.merge(votes_official_names, name_info_under_genus, left_on='official_name_id', right_on = 'name_id')

# we've whittled it down from 360K to 200K legit observation votes
# now we'll throw away votes with negative scores
# this gives us about 190K observations
votes_master = votes_master.drop(['official_name_id'], axis=1)
votes_master = votes_master.loc[votes_master['vote_cache'] > 0]

# we'll also only consider the ~1000 groups/species with top votes 
get_top_groups = votes_master.groupby('name_id').size()
get_top_groups.sort()
top_groups = get_top_groups.tail(1000)
top_groups_name_ids = top_groups.keys().tolist()

# there are 280K images from master that are in these top 1000 species/varieties/etc.
len(master.loc[master['official_name_id'].isin(top_groups_name_ids)])
# for reference, there are 244K images from master that are classified as genus or above 
# so there are only 450K species or better classified images
# and 62% of them are in the top 1000 species

# only use votes for one of the top groups
votes_master = votes_master.loc[votes_master['name_id'].isin(top_groups_name_ids)]

votes_matrix = pd.DataFrame(np.zeros((len(votes_master), len(top_groups_name_ids))), columns = top_groups_name_ids)
obs_id_to_index = {}
observations = votes_master.groupby('observation_id').groups.keys()
largest_index = 0
for obs in observations:
	obs_id_to_index[obs] = largest_index
	largest_index += 1

# add votes to the vote matrix
count = 0
for key, row in votes_master.iterrows():
	matrix_ind = obs_id_to_index[row['observation_id']]
	### HERE WE SET THE VOTE TO BE 1 OR VOTE_CACHE!!
	votes_matrix.iloc[matrix_ind][row['name_id']] = 1.0
	count += 1

# write the votes matrix to file
votes_matrix.to_csv(path_or_buf='votes_matrix_for_CF_0or1.csv', sep='\t')

votes_matrix = pd.read_csv('votes_matrix_for_CF_0or1.csv', sep='\t')

# create item-item matrix
item_item = pd.DataFrame(index=votes_matrix.columns,columns=votes_matrix.columns)
# Lets fill in those empty spaces with cosine similarities
# Loop through the columns
count = 0
for i in top_groups_name_ids :
    print(count)
    count += 1
    # Loop through the columns for each column
    for j in top_groups_name_ids :
      # Fill in placeholder with cosine similarities
      item_item.ix[i,j] = 1-cosine(votes_matrix.ix[:,i],votes_matrix.ix[:,j])

item_item.to_csv(path_or_buf='similarity_matrix_for_CF_0_or_1.csv', sep='\t')
item_item = pd.read_csv('similarity_matrix_for_CF_0_or_1.csv', sep='\t') 


# now we have 126K votes

'''
METHOD FOR MERGING CLUSTERS
1. dataframe item_item: columns = (bucket1_id, bucket2_id, sim_score)
2. dict-A bucket_ind_to_species_ids: bucket_1: (1, 2, 3), bucket_2: (4, 5, 6)
3. dict-B species_id_to_bucket_ind: 1: bucket_2, 2: bucket_3
4. votes matrix

1. sort df by score: (1,2) is highest
2. merge bucket2 into bucket1:
	a. update bucket1: (1,2) in dict-A
	b. remove bucket2 in dict-A
	b. update species2 to = bucket1 in dict-B
3. update votes matrix
	a. vm: remove column for bucket 2
	b. vm: merge votes from bucket 1 into bucket1
4. update sim scores
	a. df: remove all sim scores for bucket2 
	b. df: update all sim scores that include bucket1  
'''

# 1. Constructing df of similarities
votes_df = pd.DataFrame(columns=['bucket1', 'bucket2', 'sim'])
count =0
for key, row in item_item.iterrows():
	print(count)
	count += 1
	for i in row.index:
		if key < i:
			votes_df = votes_df.append({'bucket1': key, 'bucket2': i, 'sim': row[i]}, ignore_index=True)

votes_df.to_csv(path_or_buf='flattened_similarity_dataframe_for_CF_0_or_1.csv', sep='\t')

votes_df =pd.read_csv('flattened_similarity_dataframe_for_CF_0_or_1.csv', sep='\t')


# 2. constructing bucket_ind_to_species_ind dict
# initially set every species ind to a bucket ind
bucket_to_species_dict = {}
for i in top_groups_name_ids :
	bucket_to_species_dict[i] = [i]

# 3. constructing species_ind_to_bucket_ind dict
# initially set every species ind to a bucket ind
species_to_bucket_dict = {}
for i in top_groups_name_ids :
	species_to_bucket_dict[i] = i

# NOW, start iterative process

steps = 50
sim_limit = 0.025
for i in range(steps):
	print("MERGE-STEP: " + str(i))
	# step 1 : sort the sim DF to find the highest similarity buckets
	votes_df = votes_df.sort('sim', ascending=False)
	# step 2: update the bucket/species dictionaries and merging
	merged_into_ind = votes_df.head(1)['bucket1'].values.tolist()[0]
	merging_ind = votes_df.head(1)['bucket2'].values.tolist()[0]
	sim = votes_df.head(1)['sim'].values.tolist()[0]
	if(merged_into_ind == merging_ind):
		print "ERROR!!"
		break
	if(sim < sim_limit):
		print "REACHED SIM LIMIT"
		break
	bucket2_size = len(bucket_to_species_dict[merging_ind])
	bucket1_size = len(bucket_to_species_dict[merged_into_ind])
	species_to_merge = bucket_to_species_dict.pop(merging_ind)
	species_merged_with = bucket_to_species_dict[merged_into_ind]
	names_species_to_merge = [name_info_under_genus.loc[name_info_under_genus['name_id']==x]['text_name'].values.tolist()[0] for x in species_to_merge]
	names_species_merged_with = [name_info_under_genus.loc[name_info_under_genus['name_id']==x]['text_name'].values.tolist()[0] for x in species_merged_with]
	print("Step 1: With score " + str(sim) + ", Merging index  "+ str(merging_ind) + " with species " +  ",".join(names_species_to_merge) + " into index " + str(merged_into_ind) + " with species " + ",".join(names_species_merged_with))
	old = bucket_to_species_dict[merged_into_ind]
	old.extend(species_to_merge)
	bucket_to_species_dict[merged_into_ind] = old
	for item in species_to_merge:
		species_to_bucket_dict[item] = merged_into_ind
	print("Step 2: Updated dictionaries")
	# step 3 update votes matrix by moving votes for bucket 2 into bucket 1 and taking a weighted average. 
	# weighted average calc: (new_vote * (bucket2_size) + old_vote * (bucket1_size)) / (bucket1_size + bucket2_size)
	votes_to_move = votes_matrix.ix[:,merging_ind]
	del votes_matrix[merging_ind]
	votes_matrix.ix[:,merged_into_ind] = ((votes_matrix.ix[:,merged_into_ind] * bucket1_size) + (votes_to_move * bucket2_size))/(bucket1_size + bucket2_size)
	print("Step 3: Updated vote matrix")
	# step 4 update sim scores
	votes_df = votes_df[votes_df['bucket1']!= merging_ind]
	votes_df = votes_df[votes_df['bucket2']!= merging_ind]
	# update sim scores for entry where one bucket is bucket1 (merged into bucket) 
	votes_df.ix[votes_df.bucket1==merged_into_ind, 'sim'] = votes_df.ix[votes_df.bucket1==merged_into_ind, 'bucket2'].apply(lambda x: get_cosine(x, merged_into_ind))
	votes_df.ix[votes_df.bucket2==merged_into_ind, 'sim'] = votes_df.ix[votes_df.bucket2==merged_into_ind, 'bucket1'].apply(lambda x: get_cosine(x, merged_into_ind))
	print("Step 4: Updated sim scores")

bucket_names = [[name_info.loc[name_info['name_id'] == x]['text_name'].values.tolist()[0] for x in y] for y in bucket_to_species_dict.values()]
species_sizes = pd.DataFrame({'count' : master.groupby( [ "text_name"] ).size()}).reset_index()
buckets_with_sizes = []
for bucket in bucket_names:
	bucket_size = 0
	for species in bucket:
		#if species == 'Amanita solaniolens':
			#print "HERRE!", bucket
		#print species, species_sizes.loc[species_sizes['text_name'] == species]['count']
		bucket_size += species_sizes.loc[species_sizes['text_name'] == species]['count'].values.tolist()[0]
	buckets_with_sizes.append([bucket, bucket_size])
buckets_with_sizes = sorted(buckets_with_sizes,key=lambda x: x[1], reverse=True)

# remove 'mixed collection'
buckets_with_sizes = [x for x in buckets_with_sizes if 'Mixed collection' not in x[0]]
top_300_buckets = buckets_with_sizes[0:300]

def get_cosine(bucket2_id, merged_into_ind):
	return 1-cosine(votes_matrix.ix[:,merged_into_ind],votes_matrix.ix[:,bucket2_id])

## DID A BUNCH OF PROCESSING IN EXCEL. LOAD IT BACK
visual_tax = pd.read_csv('results/visual_taxonomy.txt', sep='\t')
visual_tax['Group'] = visual_tax['Group'].apply(lambda x: x.split(','))
visual_tax['Group'] = visual_tax['Group'].apply(lambda x: [y.replace("[", "").replace("]", "").replace("\'", "").strip(" ") for y in x])
visual_tax['Bucket_id'] = visual_tax.index
name_id_to_bucket_dict = {}
name_and_bucket_df = pd.DataFrame(columns = ['name_id', 'bucket_id'])
for index, bucket in visual_tax.iterrows():
	for name in bucket['Group']:
		get_name = name_info_under_genus.loc[name_info_under_genus['text_name'] == name]
		if(len(get_name) < 1):
			print(name +  " ERROR")
		else:
			to_append_name = get_name['name_id'].values.tolist()[0]
			name_id_to_bucket_dict[to_append_name] = bucket['Bucket_id']
			name_and_bucket_df = name_and_bucket_df.append({'name_id': to_append_name, 'bucket_id': bucket['Bucket_id']}, ignore_index=True)

# 227K images here, as expected
master_in_top_buckets = pd.merge(master, name_and_bucket_df, left_on = "official_name_id", right_on = "name_id")
master_in_top_buckets['img_url'] = 'http://images.mushroomobserver.org/320/' +  master_in_top_buckets['image_id'].apply(lambda x: str(x)) + '.jpg'
master_in_top_buckets.to_csv(path_or_buf='images_in_top_300_buckets_iterative_clustering_manual0.csv', sep='\t')
## GET IMAGES FOR BUCKETS
##

