import os, sys
import argparse

import tensorflow as tf

# Load trained model and evaluate images 
# Generate top 5 predictions per image and write to file

parser = argparse.ArgumentParser(description='Label test images')
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
    outfile = open("species_model/full_data_predictions.txt", "w")

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
# the following lets you get all image paths from a file
# images_path = sys.argv[1]
# f = open(images_path, "r")
# images = f.read().split('\n')
# eval_images(images)
 
def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r

#the following lets you get all images in a dir
all_images_paths = list_files('species_data')
eval_images(all_images_paths)



