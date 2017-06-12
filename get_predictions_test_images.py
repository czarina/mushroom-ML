import os, sys
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def eval_images(images_path):
    

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line 
                       in tf.gfile.GFile("species_model/retrained_labels.txt")]

    # Unpersists graph from file
    with tf.gfile.FastGFile("species_model/retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    # write predictions to file
    outfile = open("species_model/test_predictions.txt", "w")

    with tf.Session() as sess:
        for image_path in images_path:
            # Read in the image_data
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()
            # Feed the image_data as input to the graph and get first prediction
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            
            predictions = sess.run(softmax_tensor, \
                     {'DecodeJpeg/contents:0': image_data})
            
            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            
            # Print top 5 predictions per image
            results = []
            for node_id in top_k[0:5]:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                results.append(human_string + "," + str(round(score,4)))
            outfile.write(image_path + '\t' + '\t'.join(results) + '\n')
            print(image_path + '\t' + '\t'.join(results) + '\n')

# change this as you see fit
images_path = sys.argv[1]
f = open(images_path, "r")
images = f.read().split('\n')
eval_images(images)

# get per-species accuracy

def eval_per_class_accuracy():
	f = open('../results/species_model/test_predictions.txt', "r")
	test_data = f.read().split('\n')
	test_data = [x.split('\t') for x in test_data]
	df = pd.DataFrame(test_data, columns = ["img", "pred1", "pred2", "pred3", "pred4", "pred5"])
	df= df.drop(df.index[20292])
	df['pred1'] = df['pred1'].apply(lambda x: get_pred_dict(x))
	df['pred2'] = df['pred2'].apply(lambda x: get_pred_dict(x))
	df['pred3'] = df['pred3'].apply(lambda x: get_pred_dict(x))
	df['pred4'] = df['pred4'].apply(lambda x: get_pred_dict(x))
	df['pred5'] = df['pred5'].apply(lambda x: get_pred_dict(x))
	df['gold_label'] = df['img'].apply(lambda x: x[26:x.rfind("/")].replace("_", " ").capitalize())

	# some names got messed up

	df.ix[df.gold_label=='Boletus rex-veris', 'gold_label'] = 'Boletus rex veris'
	df.ix[df.gold_label=='Laccaria amethysteo-occidentalis', 'gold_label'] = 'Laccaria amethysteo occidentalis'
	

	df['top_1_is_match'] = df.apply(lambda row: 1 if row['gold_label'] == row['pred1']['name'] else 0, axis=1)
	df['top_5_is_match'] = df.apply(lambda row: 1 if row['gold_label'] in [row['pred1']['name'],row['pred2']['name'],row['pred3']['name'],row['pred4']['name'],row['pred5']['name']] else 0,  axis=1)
	# calculate number of true positives, and TP rate, in top-1 and top-5 per species
	top1_matches = df.groupby(['gold_label'])['top_1_is_match'].sum().astype(float)
	top1_test_size = df.groupby(['gold_label'])['top_1_is_match'].size().astype(float)
	top1_rate_TP = top1_matches/top1_sizes
	top5_matches = df.groupby(['gold_label'])['top_5_is_match'].sum()
	top5_sizes = df.groupby(['gold_label'])['top_5_is_match'].size().astype(float)
	top5_rate_TP = top5_matches/top5_sizes
	per_species_performance = pd.concat([top1_matches, top1_sizes, top1_rate_TP, top5_sizes, top5_rate_TP], axis=1).reset_index()
	per_species_performance.columns=['species', 'top1_matches', 'test_examples', 'top1_true_positive_rate', 'top5_matches', 'top5_true_positive_rate']
	per_species_performance = per_species_performance.sort('top5_true_positive_rate', ascending=False)
	per_species_performance.ix[per_species_performance.species == "Laccaria amethysteo-occidentalis", "species"] = "Laccaria amethysteo occidentalis"
	per_species_performance.ix[per_species_performance.species == "Boletus rex-veris", "species"] = "Boletus rex veris"


	# plot ECDF of top-1 /top-5 true positive rate (sensitivity)
	from scipy import stats
	from matplotlib import pyplot as plt

	top1_sensitivity = per_species_performance['top1_true_positive_rate'].values.tolist()
	plot_ecdf(top1_sensitivity, 'Per-species Top-1 sensitivity ECDF (Top 500 Species)', 'Species Top-1 True Positive Rate')
	top5_sensitivity = per_species_performance['top5_true_positive_rate'].values.tolist()
	plot_ecdf(top5_sensitivity, 'Per-species Top-5 sensitivity ECDF (Top 500 Species)', 'Species Top-5 True Positive Rate')

	# plot top-5 TP rate vs. # test examples
	fig, ax = plt.subplots(1, 1)
	ax.hold(True)
	ax.scatter(per_species_performance['test_examples'].values.tolist(), top5_sensitivity)
	ax.set_xlabel("Number of test examples in species")
	ax.set_ylabel("Species Top-5 True Positive Rate")
	ax.set_title("Model performance vs. species size")
	plt.show()


def plot_ecdf(species_rates, title, xaxis):
	# compute the ECDF of the samples
	qe, pe = ecdf(species_rates)

	# plot
	fig, ax = plt.subplots(1, 1)
	ax.hold(True)
	ax.plot(qe, pe, '-r', lw=2, label='Empirical CDF')
	ax.set_xlabel(xaxis)
	ax.set_ylabel('Cumulative probability')
	ax.legend(fancybox=True, loc='right')
	ax.set_title(title)

	plt.show()

def ecdf(sample):

    # convert sample to a numpy array, if it isn't already
    sample = np.atleast_1d(sample)

    # find the unique values and their corresponding counts
    quantiles, idx = np.unique(sample, return_inverse=True)
    counts = np.bincount(idx)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob

# generate a  vector of length 500 for each species. 
# v_i,j (the vector for species i in position j) = # of test examples from species i that had guesses of species j with score > 0.01
def cluster_similar_mushrooms(df):
	
	species = set(df['gold_label'].values.tolist())
	count = 0
	species_to_index_dict = {}
	# dict of species: index
	for a in species:
		species_to_index_dict[a] = count
		count = count + 1
	index_to_species_dict = [""]*500
	for key in species_to_index_dict.keys():
		index_to_species_dict[species_to_index_dict[key]] = key

	# 1. for every species i create 1 vector of length 500. v_i, j = # of number of examples in species i with predictions of species j and score > 0.01
	vectors = {}
	for index, row in df.iterrows():
		if row['gold_label'] in vectors:
			vector = vectors[row['gold_label']]
		else:
			vector = [0.0] * 500
		vector = process_prediction_for_vectors(vector, row['pred1'], species_to_index_dict)
		vector = process_prediction_for_vectors(vector, row['pred2'], species_to_index_dict)
		vector = process_prediction_for_vectors(vector, row['pred3'], species_to_index_dict)
		vector = process_prediction_for_vectors(vector, row['pred4'], species_to_index_dict)
		vector = process_prediction_for_vectors(vector, row['pred5'], species_to_index_dict)
		vectors[row['gold_label']] = vector

	# 2. For every species compute the % of its test examples that had a positive (>0.01) match with species j among top 5 guesses (normalize by test example size)
	# . Get rid of percentages less than 10% - i.e. if less than 10% of test examples guessed species j with >0.01 score, set it to 0%.
	for species, vector in vectors.iteritems():
		print species
		species_test_size = per_species_performance.loc[per_species_performance['species'] == species]['test_examples'].values[0]
		vectors[species] = [x/species_test_size for x in vectors[species]]
		vectors[species] = [x if x > 0.1 else 0 for x in vectors[species]]

	# 3 Remove species i with <2 species j with non-zero vector values
	# if a species i has 0 species j with non-zero vector values, then there is not enough info to cluster this species. Either it must be manually clustered or thrown out 
	# if a species i has 1 species j with non-zero vector values, if i == j, then i is a standalone. if i != j, then ??
	# if ... >= 2 then this is a candidate for clustering.
	species_with_0_significant_matches = []
	species_with_1_significant_match = {}
	master_vectors = {}
	for species, vector in vectors.iteritems():
		matches = sum(1 if x > 0 else 0 for x in vectors[species])
		if matches == 1:
			species_with_1_significant_match[species] = vector
		if matches > 1:
			master_vectors[species] = vector

	# this yields 261 species with 2+ matches, candidates for clustering
	# 59 species with 1 match
	# 180 species with no hope

	# write candidates to file 
	with open("species_vectors_for_clustering.csv","w") as f:
		writer = csv.writer(f, delimiter='\t', lineterminator='\n')
		for species, vector in master_vectors.iteritems():
			writer.writerow([species] + vector)

	# create tuples of non-identical matches (species i, species j)
	matches = pd.DataFrame({'species': [], 'prediction_species': [], 'match_frac': []})

	for species, vector in master_vectors.iteritems():
		vector_dict = {}
		for i in range(len(vector)):
			if vector[i] > 0:
				prediction = index_to_species_dict[i]
				if prediction != species:
				#print "hi"
					matches = matches.append([{'species': species, 'prediction_species': prediction, 'match_frac': vector[i]}])
	
	# sort by frac
	matches = matches.sort('match_frac', ascending=False)

	# start merging species starting from the highest match fracs
	species_buckets = [[x] for x in  set(df['gold_label'].values.tolist())]
	count = 0
	for index, row in matches.iterrows():
		if count > 100:
			break
		count = count + 1
		ind_i_matching = [1 if row['species'] in x else 0 for x in species_buckets]
		ind_i = ind_i_matching.index(1)
		ind_j_matching = [1 if row['prediction_species'] in x else 0 for x in species_buckets]
		ind_j = ind_j_matching.index(1)
		if ind_i != ind_j:
			larger_ind = max(ind_i, ind_j)
			smaller_ind = min(ind_i, ind_j)
			to_merge = species_buckets.pop(larger_ind)
			merged_into = species_buckets[smaller_ind]
			print("Pair: " + row['species'] + "," + row['prediction_species'] + "... Merging " + ",".join(to_merge) + " into " + ",".join(merged_into))
			species_buckets[smaller_ind].extend(to_merge)


	# if a prediction has a score > 0.01, increment the tally for the predicted name by 1 
def process_prediction_for_vectors(vector, pred, species_to_index_dict):
	if pred['score'] > 0.01:
		prediction = pred['name']
		index_to_increment = species_to_index_dict[prediction]
		vector[index_to_increment] += 1
	return vector

def mean(a):
    return sum(a) / len(a)

def get_pred_dict(combined_pred):
	return {"name": combined_pred.split(",")[0].capitalize(), "score": float(combined_pred.split(',')[1])}
