import csv
import shutil,os
from pathlib import Path

    
import tensorflow as tf
import sys

print(os.getcwd())
    # Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
               in tf.gfile.GFile(os.getcwd() + "/training/inception/retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile(os.getcwd() +"/training/inception/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')


    for i in range(1,11):
        if(i<6):
            print('--img_'+str(i)+"-->Actual image of: Cat" )
        else:
            print('--img_'+str(i)+"-->Actual image of: Dog" )
        image_path = os.getcwd() + '/training/inception/img_'+str(i)+'.jpg'
    
            # Read in the image_data
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        predictions = sess.run(softmax_tensor, \
            {'DecodeJpeg/contents:0': image_data})
    
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        
        print (label_lines[0]+": confidence : "+str(predictions[0][0]));
        print (label_lines[1]+": confidence :"+str(predictions[0][1]));
        print("")
