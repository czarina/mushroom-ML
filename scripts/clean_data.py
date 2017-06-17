#if top-2 predictions of an image both appear with a frequency <2% among top-1 preds in the species, discard
#it's bad
import pandas as pd
import numpy as np
import csv
from scipy.spatial.distance import cosine
import subprocess
import os

def discard_bad_images():

  df = pd.read_csv('results/species_model/full_data_predictions.txt', sep='\t', header=None)
  df.columns = ['img', 'pred1', 'pred2', 'pred3', 'pred4', 'pred5']
  df['pred1'] = df['pred1'].apply(lambda x: get_pred_dict(x))
  df['pred1_name'] = df['pred1'].apply(lambda x: x['name'])
  df['pred2'] = df['pred2'].apply(lambda x: get_pred_dict(x))
  df['pred2_name'] = df['pred2'].apply(lambda x: x['name'])
  df['pred3'] = df['pred3'].apply(lambda x: get_pred_dict(x))
  df['pred3_name'] = df['pred3'].apply(lambda x: x['name'])
  df['pred4'] = df['pred4'].apply(lambda x: get_pred_dict(x))
  df['pred4_name'] = df['pred4'].apply(lambda x: x['name'])
  df['pred5'] = df['pred5'].apply(lambda x: get_pred_dict(x))
  df['pred5_name'] = df['pred5'].apply(lambda x: x['name'])
  df['gold_label'] = df['img'].apply(lambda x: x[13:x.rfind("/")].replace("_", " ").capitalize())

  # some names got messed up

  df.ix[df.gold_label=='Boletus rex-veris', 'gold_label'] = 'Boletus rex veris'
  df.ix[df.gold_label=='Laccaria amethysteo-occidentalis', 'gold_label'] = 'Laccaria amethysteo occidentalis'
  

  df['top_1_is_match'] = df.apply(lambda row: 1 if row['gold_label'] == row['pred1']['name'] else 0, axis=1)
  df['top_5_is_match'] = df.apply(lambda row: 1 if row['gold_label'] in [row['pred1']['name'],row['pred2']['name'],row['pred3']['name'],row['pred4']['name'],row['pred5']['name']] else 0,  axis=1)
 
  # get top1 guesses and %s per species 
  grouped_df = df.groupby( [ "gold_label", "pred1_name"] )
  gold_and_preds_count = pd.DataFrame(grouped_df.size().reset_index(name = "gold_and_pred_count")) 
  examples = df.groupby(['gold_label']).size()
  example_counts = pd.DataFrame({'gold_label':examples.index, 'num_imgs':examples.values})
  merged = pd.merge(gold_and_preds_count, example_counts, left_on='gold_label', right_on='gold_label')
  merged.gold_and_pred_count = merged.gold_and_pred_count.astype(float)
  merged['perc_of_species_guessed_this_pred'] = merged['gold_and_pred_count'] / merged['num_imgs']
  top1_common_1pct_guesses = merged.loc[merged['perc_of_species_guessed_this_pred'] > 0.01]

  #(i,j) = # of examples where species j was a top5 guess for species i
  pairwise_appearance_counts = {}
  count = 0
  for i, row in df.iterrows():
    print(count)
    count += 1
    update_dict((row['gold_label'], row['pred1_name']))
    update_dict((row['gold_label'], row['pred2_name']))
    update_dict((row['gold_label'], row['pred3_name']))
    update_dict((row['gold_label'], row['pred4_name']))
    update_dict((row['gold_label'], row['pred5_name']))

  # (i,j) = % of examples where species j is a top 5 pred for species i
  # these are the rates of appearance of every species j among top 5 preds of species i
  pairwise_appearance_rate = {}
  count = 0
  for key, count in pairwise_appearance_counts.iteritems():
    gold = key[0]
    num_gold_examples = float(example_counts.loc[example_counts['gold_label'] == gold]['num_imgs'].values[0])
    rate = count / num_gold_examples
    pairwise_appearance_rate[key] = rate

  # if we use a 2.5% cut-off for the top 2 preds of each image, we get:
  bad_imgs = []
  count = 0
  start_from = 0
  for key, row in df.iterrows():
    print(count)
    count += 1
    if count > start_from:
      curr_gold = row['gold_label']
      # check if this gold, 1st pred pair is rare
      if pairwise_appearance_rate[(curr_gold, row['pred1_name'])] < 0.025 and pairwise_appearance_rate[(curr_gold, row['pred2_name'])] < 0.025:
        # pair is bad, throw out
          bad_imgs.append(row['img'])
          if not os.path.exists("bad_imgs_twoptfive_perc_top_2/" + curr_gold.replace(" ", "_")):
            os.makedirs("bad_imgs_twoptfive_perc_top_2/" + curr_gold.replace(" ", "_"))
          cmd = 'scp -i ../mushroom2.pem ubuntu@54.215.232.58:/tf_files/' + row['img'] + ' bad_imgs_twoptfive_perc_top_2/' + curr_gold.replace(" ", "_")
          print(str(count), cmd)
          try:
            result = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
          # some examples were moved to separate dir to reduce # of training examples
          except subprocess.CalledProcessError as e:
            print("ERROR!", e)
      else:
        #store in another dir if img is good. just for sanity checking.
          if not os.path.exists("good_imgs_twoptfive_perc_top_2/" + curr_gold.replace(" ", "_")):
            os.makedirs("good_imgs_twoptfive_perc_top_2/" + curr_gold.replace(" ", "_"))
          cmd = 'scp -i ../mushroom2.pem ubuntu@54.215.232.58:/tf_files/' + row['img'] + ' good_imgs_twoptfive_perc_top_2/' + curr_gold.replace(" ", "_")
          print(str(count), cmd)
          try:
            result = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
          # some examples were moved to separate dir to reduce # of training examples
          except subprocess.CalledProcessError as e:
            print("ERROR!", e)      

  # We have top 500 species among gold labels here, and a total of 200K images
  # The % of images whose 1st and 2nd preds are both <2% common among 1st preds of this species is:
  bad_imgs = []
  count = 0
  start_from = 0
  for key, row in df.iterrows():
    print(count)
    count += 1
    if count > start_from:
      curr_gold = row['gold_label']
      # check if this gold, 1st pred pair is rare
      if is_not_row_in_df(curr_gold, row['pred1_name']) and is_not_row_in_df(curr_gold, row['pred2_name']):
        # pair is bad, throw out
          bad_imgs.append(row['img'])
          if not os.path.exists("bad_imgs_onepct_top_2/" + curr_gold.replace(" ", "_")):
            os.makedirs("bad_imgs_onepct_top_2/" + curr_gold.replace(" ", "_"))
          cmd = 'scp -i ../mushroom2.pem ubuntu@54.215.232.58:/tf_files/' + row['img'] + ' bad_imgs_onepct_top_2/' + curr_gold.replace(" ", "_")
          print(str(count), cmd)
          try:
            result = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
          # some examples were moved to separate dir to reduce # of training examples
          except subprocess.CalledProcessError as e:
            print("ERROR!", e)
      else:
        #store in another dir if img is good. just for sanity checking.
          if not os.path.exists("good_imgs_onepct_top_2/" + curr_gold.replace(" ", "_")):
            os.makedirs("good_imgs_onepct_top_2/" + curr_gold.replace(" ", "_"))
          cmd = 'scp -i ../mushroom2.pem ubuntu@54.215.232.58:/tf_files/' + row['img'] + ' good_imgs_onepct_top_2/' + curr_gold.replace(" ", "_")
          print(str(count), cmd)
          try:
            result = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
          # some examples were moved to separate dir to reduce # of training examples
          except subprocess.CalledProcessError as e:
            print("ERROR!", e)      




  less_0point5_perc_guesses.to_csv(path_or_buf='results/species_model/rare_prediction_images_bad_img_cleanup.csv', sep='\t')



  # retrieve 10% of bad imgs from EC2 for inspection

  count = 0
  for bad_img in bad_imgs:
    count += 1
    if count % 10 == 0:
      cmd = 'scp -i ../mushroom2.pem ubuntu@54.215.232.58:/tf_files/' + bad_img + 'bad_images_cleanup_0point5thres/'
      print(str(count/10), cmd)
      try:
        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
      # some examples were moved to separate dir to reduce # of training examples
      except subprocess.CalledProcessError as e:
        print("ERROR!", e)

def update_dict(pair):
  if fropair in pairwise_appearance_counts:
    pairwise_appearance_counts[pair] += 1
  else:
    pairwise_appearance_counts[pair] = 1

def is_not_row_in_df(curr_gold, curr_pred):
  return len(top1_common_1pct_guesses.loc[(top1_common_1pct_guesses['gold_label']== curr_gold) & (top1_common_1pct_guesses['pred1_name'] == curr_pred)])== 0

def get_pred_dict(combined_pred):
  return {"name": combined_pred.split(",")[0].capitalize(), "score": float(combined_pred.split(',')[1])}