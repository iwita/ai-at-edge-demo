#!/usr/bin/python3


# Changes from Near
# 1. logging.info -> app.logger.info
# 2. Use of logging.basicConfig(level=logging.DEBUG)
# 3. Formatted prints
# 4. Comments of Code sections
# 5. Globals in app

from mimetypes import init
import time
import os
from datetime import datetime
from flask import Flask, request, Response
import json
import numpy as np
import cv2
from ctypes import *
from typing import List
import numpy as np
import vart
import pathlib
import xir
import threading
import sys
import argparse
import io
import shutil
import tensorflow as tf
import zipfile
import logging

from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import torch

# Is the kernel already initialized?
initialized=False

logging.basicConfig(level=logging.DEBUG) #is this needed for logging?

# This variable is the model
graph_func = None

divider = '------------------------------------'

def init_kernel(model_path):
    st=datetime.now()
    global graph_func
    #Load Model
    tensorRT_model_loaded = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])
    graph_func = tensorRT_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    # Useful for Debugging
    signature_keys = list(tensorRT_model_loaded.signatures.keys())
    app.logger.info("{}".format(signature_keys))
    app.logger.info("{}".format(graph_func.structured_outputs))
    app.logger.info("{}".format(graph_func))
    et = datetime.now()
    elasped_time_i=et-st
    app.logger.info('Initialize time:\t' + str(elapsed_time_i.total_seconds()*1000) + 'ms')
    #WARMUP = True
    #if(WARMUP == True):

def inference(indata,batch_size, model_path):
    full_start = timer()
    global graph_func
    print(divider)
    app.logger.info{"{:s}".format(divider))
    # REST API INPUT BOILERPLATE --------------------------------
    # We get the zip file with the images as Bytes
    zip_ref = zipfile.ZipFile(io.BytesIO(indata), 'r')
    zip_ref.extractall("./")
    zip_ref.close()
    # This is the name of the folder of the zip that contains all the images
    FOLDERNAME = "./ImageNet_val_folder_1000"
    listimage=os.listdir(FOLDERNAME)
    runTotal = len(listimage)
    # END OF REST API INPUT BOILERPLATE -------------------------
    # 
    # CREATE AND PREPROCESS DATASET -----------------------------
    ds_val = tf.keras.preprocessing.image_dataset_from_directory(
        directory = FOLDERNAME,
        labels = None,
        label_mode = None,
        color_mode = "rgb",
        batch_size = 1,
        image_size = (224,224),
        shuffle = False
    )
    # Preprocess
    ds_val = ds_val.batch(batch_size,drop_remainder=False)
    ds_val = ds_val.map(lambda x: preprocess(x,  model_path))
    # END OF CREATE AND PREPROCESS DATASET ----------------------
    #
    # EXPERIMENT ------------------------------------------------
    app.logger.info("-------------Experiment-------------")
    timetotal_execution=0.0
    i = 0
    app.logger.info("Dataset size: {}".format(runTotal))
    num_samples = runTotal
    iterations = num_samples // batch_size
    if(iterations * batch_size != num_samples):
        remainder_iteration = 1
    else: 
        remainder_iteration = 0
    preds = []    
    app.logger.info("Starting")
    for element in ds_val.take(iterations + remainder_iteration):
       x_test = element[0]
       if(i == iterations): # Only last iteration
           x_input = tf.zeros(shape =(batch_size, 224, 224, 3))
           x_input[0:x_test.shape[0]) = x_test[:]
       else:    
           x_input =tf.constant(x_test[:])
       start = timer()
       # Inference
       output_data = graph_func(x_input) 
       # This is really important. GPU inference run asynchronously, we need to wait for the process to end before using timer()
       # Tensorflow doesn't have this fucntionality, therefore torch.cuda.synchronize is used
       torch.cuda.synchronize()
       end = timer()
       timetotal_execution += end - start # Time in seconds, e.g. 5.38091952400282 
       if(i == iterations): # Only last iteration
           valid_outputs = x_test.shape[0]
       else:    
           valid_outputs = batch_size
       # Calculate the probs 
       for j in range(valid_outputs):
           probs = softmax(output_data[list(output_data.keys())[0]][j])
           preds.append(tf.keras.applications.imagenet_utils.decode_predictions(probs[0], top=5)[0])
       i = i + 1
    # END OF EXPERIMENT ------------------------------------------
    #
    # BENCHMARKS -------------------------------------------------
    #fps = float(runTotal / timetotal_execution)
    avg_time_execution = timetotal_execution / (iterations + remainder_iteration)
    preds = []
    for i in range(len(out_q)):
        probs = softmax(out_q[i])
        preds.append(tf.keras.applications.imagenet_utils.decode_predictions(probs[0], top=5)[0])
    # END OF BENCHMARKS ------------------------------------------
    #
    # REST API OUTPUT BOILERPLATE --------------------------------
    # Create and return dictionary with predictions
    out_dict = {}
    for i in range(len(preds)):
        out_dict[listimage[i]] = preds[i]
    # END OF REST API OUTPUT BOILERPLATE -------------------------
    #
    # PRINTS -----------------------------------------------------
    full_end = timer()
    full_time = full_end - full_start
    avg_full_time = full_time/ (iterations + remainder_iteration)
    app.logger.info('Processing Latency: (data preparation + execution):\t%.2fms (%.2f + %.2f)', avg_full_time*1000, (avg_full_time - avg_time_execution)*1000, avg_time_execution*1000)
    app.logger.info('Total throughput (batch_size) in outputs per second:\t\t%.2fps (%d)', runTotal/avg_full_time, batch_size)
    # END OF PRINTS ----------------------------------------------
    # Return Dictionary
    return out_dict

def preprocess(x_test, MODEL_PATH):
   # Preprocess images
   if "ResNet50" in MODEL_PATH:
      x_test = tf.keras.applications.resnet50.preprocess_input(x_test)
   elif "NASNet_large" in MODEL_PATH:
      x_test = tf.image.resize(x_test, (331, 331))
      x_test = tf.keras.applications.nasnet.preprocess_input(x_test)
   elif "MobileNet" in MODEL_PATH:
      x_test = tf.cast(x=x_test, dtype=tf.float32)/127.5 - 1.0
   elif "LeNet5" in MODEL_PATH:
      x_test = tf.image.rgb_to_grayscale(x_test)
      x_test = tf.cast(x=x_test, dtype=tf.float32)/255.0
   elif ("ResNetV2152" in MODEL_PATH) or ("ResNet152" in MODEL_PATH) or ("InceptionV4" in MODEL_PATH):
      x_test = tf.image.resize(x_test, (299, 299))
      x_test = tf.cast(x=x_test, dtype=tf.float32)/127.5 - 1.0
   return x_test

def softmax(logits):
   # Works for 1D np arrays
   scores = logits - np.max(logits)
   probs = np.exp(scores.astype(np.float64))/np.sum(np.exp(scores.astype(np.float64)))
   return probs

app=Flask(__name__)

@app.route('/api/infer',methods=['POST'])
def test():
    MODEL_PATH = "ResNet50_ImageNet_70_90_7_76GF_TensorRT_INT8_BATCH_128"
    BATCH_SIZE = 128
    r = request
    global initialized
    # Check if this is the first run
    if not initialized:
        init_kernel(MODEL_PATH) # This changes from case to case
        print("init")
        app.logger.info("init")
        initialized=True
    file = r.files['archive']
    file_like_object = file.read()
    # Call the service here
    print("inference")
    app.logger.info("inference")
    
    preds = inference(file_like_object, BATCH_SIZE, MODEL_PATH) # This changes from case to case

    # Return the dicitonary in json form
    return Response(response=json.dumps(preds),status=200,mimetype="application/json")

@app.route('/api/test2',methods=['POST'])
def test2():

    return Response(response=json.dumps({"res":"ok"}),status=200,mimetype="application/json")


app.run(host="0.0.0.0",port=3000) # This changes each time depending on the experiment
