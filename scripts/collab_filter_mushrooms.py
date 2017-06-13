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
name_info_under_genus = name_info.loc[name_info['rank'] < 9]


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

# now we have 126K votes

# initialize the matrix for CF
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
	votes_matrix.iloc[matrix_ind][row['name_id']] = row['vote_cache']
	count += 1

# write the votes matrix to file
votes_matrix.to_csv(path_or_buf='votes_matrix_for_CF.csv', sep='\t')

votes_matrix = pd.read_csv('votes_matrix_for_CF.csv', sep='\t')

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

# Create a placeholder items for closes neighbours to an item
data_neighbours = pd.DataFrame(columns=['name_id1', 'name_id2', 'name1', 'name2', 'score'])

# Loop through our similarity dataframe and fill in neighbouring item names
for i in range(0,len(top_groups_name_ids)):
	top_10 = item_item.ix[0:,i].order(ascending=False)[:10]
	top_over_thresh = top_10[top_10>0.01]
	top_no_match = top_over_thresh[[j != top_groups_name_ids[i]  for j in top_over_thresh.index]]
	#print indices
	#print top_groups_name_ids[i], top_no_match 
	indices= top_no_match.index.values.tolist()
	if len(indices)>0:
		for x in range(len(indices)):
			name1 = name_info_under_genus.loc[name_info_under_genus['name_id']==top_groups_name_ids[i]]['text_name'].values.tolist()[0]
			name2 = name_info_under_genus.loc[name_info_under_genus['name_id']==indices[x]]['text_name'].values.tolist()[0]
			data_neighbours = data_neighbours.append({'name_id1': top_groups_name_ids[i], 'name_id2': indices[x], 'name1': name1, 'name2': name2, 'score': top_no_match[x]}, ignore_index=True)
	#if top_groups_name_ids[i] == 949:

# sort similar pairings by score
sorted_pairs = data_neighbours.sort('score', ascending=False)
sorted_pairs = sorted_pairs[sorted_pairs.name1 != 'Mixed collection']
sorted_pairs = sorted_pairs[sorted_pairs.name2 != 'Mixed collection']
# Now start merging similar species
species_buckets = [[name_info_under_genus.loc[name_info_under_genus['name_id']==x]['text_name'].values.tolist()[0]] for x in votes_matrix.columns.values.tolist()]
count = 0
# with a cutoff score of 0.02, 755 "pairings" were identified
for index, row in sorted_pairs.iterrows():
	if row['score'] < 0.015:
		break
	count = count + 1
	ind_i_matching = [1 if row['name1'] in x else 0 for x in species_buckets]
	ind_i = ind_i_matching.index(1)
	ind_j_matching = [1 if row['name2'] in x else 0 for x in species_buckets]
	ind_j = ind_j_matching.index(1)
	if ind_i != ind_j:
		larger_ind = max(ind_i, ind_j)
		smaller_ind = min(ind_i, ind_j)
		to_merge = species_buckets.pop(larger_ind)
		merged_into = species_buckets[smaller_ind]
		#print(row['name1'] + " MATCHED with " + row['name2'] + " with score " + str(row['score']))
		#
		if 'Amanita solaniolens' in to_merge or 'Amanita solaniolens' in merged_into:
			print(",".join(to_merge) + " into " + ",".join(merged_into))
		#print("Pair: " + str(row['name_id1']) + "," + str(row['name_id2']) + " with " + str(row['score']) + " match... Merging " + ",".join(to_merge) + " into " + ",".join(merged_into))
		species_buckets[smaller_ind].extend(to_merge)

# 152 multi-item buckets were created, containing 417 species (or groups/clades/vars/whatever)
# as a result, the total number of buckets went from 1K to 735
merged_buckets = [x for x in species_buckets if len(x) > 1]

# get all buckets containing a genus
 #[y for y in species_buckets if len([x for x in y if 'Morchella' in x])>0]

# get the number of examples per bucket
species_sizes = pd.DataFrame({'count' : master.groupby( [ "text_name"] ).size()}).reset_index()
buckets_with_sizes = []
for bucket in species_buckets:
	bucket_size = 0
	for species in bucket:
		#if species == 'Amanita solaniolens':
			#print "HERRE!", bucket
		print species, species_sizes.loc[species_sizes['text_name'] == species]['count']
		bucket_size += species_sizes.loc[species_sizes['text_name'] == species]['count'].values.tolist()[0]
	buckets_with_sizes.append([bucket, bucket_size])
# sort new buckets by size
# the top 300 buckets contain 621 species/varieties/groups/whatever
# and comprise 224K examples (which is more than the top 500 species used previously)
# there are 450K total species or better classified images in master
# so we are using roughly 50% of them if we use the top 300 buckets
# the 300th bucket contains the same number of examples ~270 as the 300th species WITHOUT BUCKETING
buckets_with_sizes = sorted(buckets_with_sizes,key=lambda x: x[1], reverse=True)
top_300_buckets = buckets_with_sizes[0:300]
bucket_dictionary = {}
for i in range(len(top_300_buckets)):
	bucket_dictionary[i] = top_300_buckets[i][0]

# WRITE BUCKET DICT TO FILE
import json

with open('bucket_dict_using_0_or_voteScore.txt', 'w') as file:
     file.write(json.dumps(bucket_dictionary))

all_names_in_top_300 = []
for bucket in top_300_buckets:
	all_names_in_top_300.extend(bucket[0])
# WRITE MASTER SET OF IMAGES IN TOP 300 BUCKETS WITH IMG URLS
# 224K images here
master_images_in_top_300 = master.loc[master['text_name'].isin(all_names_in_top_300)]
master_images_in_top_300['img_url'] = 'http://images.mushroomobserver.org/320/' +  master_images_in_top_300['image_id'].apply(lambda x: str(x)) + '.jpg'
master_images_in_top_300['bucket_id'] = master_images_in_top_300['text_name'].apply(lambda x: get_bucket_id(x, top_300_buckets))
master_images_in_top_300.to_csv(path_or_buf='images_in_top_300_buckets_0orvotecache.csv', sep='\t')


def get_bucket_id(text_name, top_300_buckets):
	return [True if text_name in bucket[0] else False for bucket in top_300_buckets].index(True)
1, 3
2
3
4
5
6

1, 3
2, 3
4, 5
2, 4


# --- End Item Based Recommendations --- #
