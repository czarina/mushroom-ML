# read input CSV from MO

import pandas as pd
import numpy as np

# read MO data
locations = pd.read_csv('christina-locations.csv', sep='\t')
images = pd.read_csv('christina-images.csv', sep='\t')
names = pd.read_csv('christina-names.csv', sep='\t')
observations = pd.read_csv('christina-observations.csv', sep='\t')

# join into a master dataset
master = pd.merge(images, observations, left_on='observation_id', right_on='id')
master = pd.merge(master, locations, left_on='location_id', right_on='id')
master = pd.merge(master, names, left_on='name_id', right_on='id')

master = master.sort(['image_id'])

# create new lat/long columns
master = master.apply(process_location, axis=1)

# clean up the names table. get an official name for every name in the DB
names_official = cleanup_names(names)
names_official.columns = ['id_temp', 'official_name_id']

# use OFFICIAL_NAME for every image
master = pd.merge(master, names_official, left_on='name_id', right_on='id_temp')

# clean unwanted columns
master = master.drop(['id_x', 'name_id', 'id_y', 'north', 'south', 'east', 'west', 'id', 'text_name', 'author', 'rank', 'deprecated', 'synonym_id', 'correct_spelling_id', 'id_temp'], axis=1)

# get the official name info for every image
master = pd.merge(master, names, left_on='official_name_id', right_on='id')
master = master.drop(['id', 'official_name_id_y'], axis = 1)
master = master.rename(columns = {'official_name_id_x':'official_name_id'})

# get top names
top_names = master.groupby(['official_name_id']).size()
top_names.sort()

# now get most popular SPECIES. species is rank = 4
# get images with a confidence species classification. this is 418K out of 690K images, 60%
# we will use the rest for genus level classification
species_master = master.loc[master['rank'] == 4]
top_species_names = species_master.groupby(['official_name_id']).size()
top_species_names = pd.DataFrame({'official_name_id':top_species_names.index, 'num_examples':top_species_names.values})

top_species_names = pd.merge(top_species_names, names, left_on = 'official_name_id', right_on='id')
top_species_names = top_species_names.sort(['num_examples'])
top_500 = top_species_names.tail(500)
top_500.to_csv(path_or_buf='top_500.csv', sep='\t')

# attempt to extract genus from every image with a classification rank 9 (genus) or under (more specific)
# this is 650K examples
master_genus_and_below = master.loc[master['rank'] <= 9]
master_genus_and_below['genus'] = master_genus_and_below['text_name'].apply(lambda x: x.split()[0])
# group by genus
genii_agg_with_sizes = master_genus_and_below.groupby(['genus']).size()
genii_agg_with_sizes.sort()

# write to CSV
genii_agg_with_sizes.to_csv(path_or_buf='examples_per_genus.csv', sep='\t')

# there are 2K genii. the top 100 genii comprise 70% of all submissions
# top 200 genii comprise 82% of all submissions
# top 117 genii each have >1K examples
# top 200 genii have >500 examples (might not be enough)

top_100_genii = genii_agg_with_sizes.tail(100).index.values.tolist()
# there are 433k examples in the top 100 genii 
images_in_top_100_genii = master_genus_and_below[master_genus_and_below['genus'].isin(top_100_genii)]

# now we need to generate a list of URLs for the photos of those images
images_in_top_100_genii['img_url'] = 'http://images.mushroomobserver.org/320/' +  images_in_top_100_genii['image_id'].apply(lambda x: str(x)) + '.jpg'

# write all top-100 genus images to a file
images_in_top_100_genii.to_csv(path_or_buf='images_in_top_100_genii.csv', sep='\t')

# we will attempt to make a species level classifier
# if a specimen does not have a high confidence of matching a particular species, we will re-classify it with a genus classifier
# alternatively we can check if all top species predictions are in the same genus. the results would probably be similar
# what is accuracy of species level classification? 
# what is accurac of genus level classification?


# if a lat/long is provided, use that
# else use centroid of east/west and north/south
# otherwise leave it null
def process_location(x):
	print x['image_id']
	if not(np.isnan(x['lat'])) and not(np.isnan(x['long'])):
		return x
	else:
		x['lat'] = (x['north']+x['south'])/2
		x['long'] = (x['east']+x['west'])/2
	return x

# get an official name id for every name (deprecated and non-deprecated)
# for all names, official name id is minimum name id of all synonymous non-deprecated names
def cleanup_names(x):
	# A. we want to create an official_name for every name
	x['official_name_id'] = -1
	# filter for non deprecated names
	non_deprecated = x.loc[x['deprecated'] == 0]
	# 1. some non deprecated names have no synonym IDs. their official name id's are their own name id
	non_deprecated_standalones = non_deprecated[non_deprecated['synonym_id'].isnull()]
	non_deprecated_standalones['official_name_id'] = non_deprecated_standalones['id']

	
	# 2. of non deprecated names with synonym id's, get the minimum name id for every synoym id. this is necessary because some synonym id's
	# have multiple non deprecated name id's. we will always use the minimum name id as official
	non_deprecated_has_syn = non_deprecated[non_deprecated['synonym_id'].notnull()]
	non_deprecated_syn_mins_full = non_deprecated_has_syn.loc[non_deprecated_has_syn.groupby("synonym_id")["id"].idxmin()]
	non_deprecated_syn_mins = non_deprecated_syn_mins_full[['id', 'synonym_id']]
	# now we can get the minimum synonymous name id for every name with a synonym
	non_deprecated_has_syn = pd.merge(non_deprecated_has_syn, non_deprecated_syn_mins, on="synonym_id", how = "left")
	non_deprecated_has_syn['official_name_id'] = non_deprecated_has_syn['id_y']
	non_deprecated_has_syn['id'] = non_deprecated_has_syn['id_x']

	# full table of non deprecated names with their official names
	all_non_deprecated = pd.concat([non_deprecated_standalones, non_deprecated_has_syn], ignore_index=True)

	# 3. for all deprecated names, set official name id to the official name id of any synonymous non deprecated name
	# some deprecated names have no synonyms! throw these out
	deprecated = x.loc[x['deprecated'] == 1]
	deprecated_without_syn = deprecated.loc[deprecated['synonym_id'].isnull()]
	deprecated = deprecated.dropna(axis=0, subset=['synonym_id'])

	# get list of unique minimum name id's for joining with deprecated table
	unique_non_deprec = pd.concat([non_deprecated_standalones, non_deprecated_syn_mins_full])
	unique_non_deprec = unique_non_deprec[['id', 'synonym_id']]
	deprecated_names_with_non_deprecated = pd.merge(deprecated, unique_non_deprec, how='left', on='synonym_id')
	deprecated_names_with_non_deprecated['official_name_id'] = deprecated_names_with_non_deprecated['id_y']
	deprecated_names_with_non_deprecated['id'] = deprecated_names_with_non_deprecated['id_x']

	# for the deprecated names without synonyms, official name id is their name id
	deprecated_without_syn['official_name_id'] = deprecated_without_syn['id']

	all_deprecated = pd.concat([deprecated_without_syn, deprecated_names_with_non_deprecated], ignore_index=True)

	# merge deprecated and non deprecated names together. now we have an official name id for every name!
 	names_with_official = pd.concat([all_non_deprecated, all_deprecated], ignore_index=True)[['id', 'official_name_id']]

 	return names_with_official

def get_official_names()
def process_name(x):
	print x['image_id']
	if x['deprecated']==1:

