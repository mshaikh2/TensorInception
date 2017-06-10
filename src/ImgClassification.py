'''
Created on Jun 6, 2017

@author: Abuzar
'''

import tensorflow as tf
import urllib, requests, os

def download_imageNetDataset():
    DOWNLOAD_FILE_PATH = os.getcwd() + '/trainingImgs/Dog/'
    urlPathFile = os.getcwd() + '/urlDogs'
    counter = 529
    print(os.getcwd())
    with open(urlPathFile, 'r',400,'utf-8') as file:
        for line in file:
            counter += 1
            try:
                urllib.request.urlretrieve(line, DOWNLOAD_FILE_PATH + "img_" + str(counter)+".jpg")
            except Exception:
                print(str(counter)+": "+"error in url: " + line)
           
   
download_imageNetDataset()
