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
import contextlib
from datetime import datetime
from flask import Flask, request, Response
import json
import numpy as np
from ctypes import *
from typing import List
import numpy as np
import threading
import sys
import argparse
import io
import shutil
import tensorflow as tf
import zipfile
import logging
import cv2

from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import torch

WIDTH  = 224
HEIGHT = 224
#normalization factor
NORM_FACTOR = 127.5

#number of classes
NUM_CLASSES = 12

# names of classes
CLASS_NAMES = ("Sky",
               "Wall",
               "Pole",
               "Road",
               "Sidewalk",
               "Vegetation",
               "Sign",
               "Fence",
               "vehicle",
               "Pedestrian",
               "Bicyclist",
               "miscellanea")

# colors for segmented classes
colorB = [128, 232, 70, 156, 153, 153,  30,   0,  35, 152, 180,  60,   0, 142, 70, 100, 100, 230,  32]
colorG = [ 64,  35, 70, 102, 153, 153, 170, 220, 142, 251, 130,  20,   0,   0,  0,  60,  80,   0,  11]
colorR = [128, 244, 70, 102, 190, 153, 250, 220, 107, 152,  70, 220, 255,   0,  0,   0,   0,   0, 119]
CLASS_COLOR = list()
for i in range(0, 19):
    CLASS_COLOR.append([colorR[i], colorG[i], colorB[i]])
COLORS = np.array(CLASS_COLOR, dtype="float32")

# Is the kernel already initialized?
initialized=False
warmed_up = False
logging.basicConfig(level=logging.DEBUG) #is this needed for logging?

# This variable is the model
graph_func = None
divider = '------------------------------------'

def init_kernel(model_path,batch_size):
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
    elapsed_time_i=et-st
    app.logger.info('Initialize time :\t' + str(int(elapsed_time_i.total_seconds()*1000)) + ' ms')

def warm_up(batch_size):
    wm_start = datetime.now()
    global graph_func
    preds = []
    x_dummy = tf.random.normal(shape =(batch_size, 224, 224, 3))
    x_input = tf.constant(x_dummy[tf.newaxis,0,:])
    output_data = graph_func(x_input)
    torch.cuda.synchronize()
    wm_end = datetime.now()
    app.logger.info('Warmup time :\t' + str(int((wm_end-wm_start).total_seconds()*1000)) + ' ms')

def inference(indata,batch_size, model_path):
    full_start = time.time()
    global graph_func
    # REST API INPUT BOILERPLATE --------------------------------
    # Data --> Image . Assume we get the data in numpyarray of image encoded bytes
    img=cv2.imdecode(np.fromstring(indata.data,np.uint8),cv2.IMREAD_COLOR)
    runTotal = 1
    out_q = [None] * runTotal
    # END OF REST API INPUT BOILERPLATE -------------------------
    # 
    # CREATE AND PREPROCESS DATASET -----------------------------
    x_test = preprocess_image(img)
    # END OF CREATE AND PREPROCESS DATASET ----------------------
    #
    # EXPERIMENT ------------------------------------------------
    app.logger.info("Dataset size: {}".format(runTotal))
    #num_samples = runTotal
    x_input = tf.constant(x_test[tf.newaxis,:])
    start =  time.time()
    output_data = graph_func(x_input)
    torch.cuda.synchronize()
    end = time.time()
    y_pred1_i = np.argmax(output_data[list(output_data.keys())[0]], axis=3) # Expected shape is (1, HEIGHT, WIDTH) and each index is the number of the color??
    print(y_pred1_i.shape) # Check if (1, HEIGHT, WIDTH) or (HEIGHT, WIDTH)
    # END OF EXPERIMENT ------------------------------------------
    #
    # BENCHMARKS -------------------------------------------------
    timetotal_execution = end - start
    avg_time_execution = timetotal_execution/runTotal
    # END OF BENCHMARKS ------------------------------------------
    #
    # REST API OUTPUT BOILERPLATE --------------------------------
    # Create and return dictionary with predictions
    segmentated_image = give_color_to_seg_img(y_pred1_i[0], NUM_CLASSES)
    segmentated_image = (segmentated_image*255.0).astype(np.uint8)
    _, seg_img_encoded = cv2.imencode('.png', segmentated_image)
    # END OF REST API OUTPUT BOILERPLATE -------------------------
    #
    # PRINTS -----------------------------------------------------
    full_end = time.time()
    full_time = full_end - full_start
    avg_full_time = full_time / runTotal
    app.logger.info(' ')
    app.logger.info('\tProcessing Latency (data preparation + execution) :  \t%d ms (%d + %d)', int(avg_full_time*1000), int((avg_full_time - avg_time_execution)*1000), int(avg_time_execution*1000))
    app.logger.info('\tTotal throughput (batch size) :                      \t%d fps (%d)', int(runTotal/full_time), batch_size)
    # END OF PRINTS ----------------------------------------------
    # Return encoded image in string
    return seg_img_encoded

def preprocess_image(image):
    #Image normalization
    #Args:     Image
    #Returns:  normalized image
    image= image.astype(np.float32)
    image = image / NORM_FACTOR - 1.0
    # image = 
    return image

def give_color_to_seg_img(seg,n_classes):
    if len(seg.shape)==3:
        seg = seg[:,:,0]
    seg_img = np.zeros( (seg.shape[0],seg.shape[1],3) ).astype('float')
    #colors = sns.color_palette("hls", n_classes) #DB
    colors = COLORS #DB
    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( colors[c][0]/255.0 ))
        seg_img[:,:,1] += (segc*( colors[c][1]/255.0 ))
        seg_img[:,:,2] += (segc*( colors[c][2]/255.0 ))

    return(seg_img)


app=Flask(__name__)

@app.route('/api/infer',methods=['POST'])tf
def test():
    MODEL_PATH = "best_UNET_v3_TensorRT_FP16_BATCH_1"
    BATCH_SIZE = 1
    r = request
    global initialized
    global warmed_up
    # Check if this is the first run
    if not initialized:
        print("init")
        app.logger.info("init")
        init_kernel(MODEL_PATH, BATCH_SIZE) # This changes from case to case
        initialized=True
    if not warmed_up:
        print("warm_up")
        app.logger.info("warm_up")
        warm_up(BATCH_SIZE)
        warmed_up=True

    # Call the service here
    print("inference")
    app.logger.info("inference")
    seg_img_string = inference(r, BATCH_SIZE, MODEL_PATH)
    #seg_img_string = inference(np.fromstring(r.data,np.uint8), THREADS)
    # Returns np.array as string. Need to imdecode in client
    return Response(response=seg_img_string.tobytes(),status=200,mimetype="image/png") 

@app.route('/api/test2',methods=['POST'])
def test2():

    return Response(response=json.dumps({"res":"ok"}),status=200,mimetype="application/json")


app.run(host="0.0.0.0",port=3001) # This changes each time depending on the experiment

