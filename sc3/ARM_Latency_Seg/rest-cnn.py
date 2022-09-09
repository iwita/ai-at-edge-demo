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
import threading
import sys
import argparse
import io
import shutil
import tensorflow as tf
import zipfile
import logging
import cv2

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
interpreter = None
divider = '------------------------------------'

def init_kernel(model_path,batch_size, num_threads):
    st=datetime.now()
    global interpreter
    #Load Model
    interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
    print(interpreter)
    input_details = interpreter.get_input_details()
    # Get input details so we can change them and allow Batch size input
    interpreter.resize_tensor_input(input_details[0]['index'], [batch_size, input_details[0]['shape'][1], input_details[0]['shape'][2], input_details[0]['shape'][3]])
    # Allocate tensors is VITAL
    interpreter.allocate_tensors()
    # Useful for Debugging
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    app.logger.info("{}".format(input_details[0]['shape']))
    app.logger.info("{}".format(input_details[0]['dtype']))
    app.logger.info("{}".format(output_details[0]['shape']))
    app.logger.info("{}".format(output_details[0]['dtype']))
    et = datetime.now()
    elapsed_time_i=et-st
    app.logger.info('Initialize time :\t' + str(int(elapsed_time_i.total_seconds()*1000)) + ' ms')

def warm_up(batch_size):
    wm_start = datetime.now()
    global interpreter
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    x_dummy = np.zeros(shape =(batch_size, 224, 224, 3), dtype=np.uint8)
    interpreter.set_tensor(input_details[0]['index'], x_dummy)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    wm_end = datetime.now()
    app.logger.info('Warmup time :\t' + str(int((wm_end-wm_start).total_seconds()*1000)) + ' ms')

def inference(indata,batch_size, model_path):
    full_start = time.time()
    global interpreter
    # REST API INPUT BOILERPLATE --------------------------------
    # Data --> Image . Assume we get the data in numpyarray of image encoded bytes
    rest_api_input_start = time.time()
    img=cv2.imdecode(np.fromstring(indata.data,np.uint8),cv2.IMREAD_COLOR)
    runTotal = 1
    rest_api_output_end = time.time() 
    print("Rest API Input %d ms" %((rest_api_output_end - rest_api_input_start)*1000))
    # END OF REST API INPUT BOILERPLATE -------------------------
    # 
    # CREATE AND PREPROCESS DATASET -----------------------------
    details_start = time.time()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    details_end = time.time()
    print("Get input output details time %d ms" %((details_end - details_start)*1000))
    pre_start = time.time()
    x_test = preprocess_image(img, input_details[0])
    pre_end= time.time()
    print("Preprocess time %d ms" %((pre_end - pre_start)*1000))
    # END OF CREATE AND PREPROCESS DATASET ----------------------
    #
    # EXPERIMENT ------------------------------------------------
    app.logger.info("Dataset size: {}".format(runTotal))
    #num_samples = runTotal
    start =  time.time()
    interpreter.set_tensor(input_details[0]['index'], x_test[np.newaxis, :])
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    end = time.time()
    post_start = time.time()
    y_pred1_i = np.argmax(output_data, axis=3) # Expected shape is (1, HEIGHT, WIDTH) and each index is the number of the color??
    post_end = time.time()
    print("Postprocess time %d ms" %((post_end - post_start)*1000))
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
    color_start = time.time()
    segmentated_image = give_color_to_seg_img(y_pred1_i[0], NUM_CLASSES)
    color_end = time.time()
    encode_start = time.time()
    segmentated_image = (segmentated_image*255.0).astype(np.uint8)
    _, seg_img_encoded = cv2.imencode('.png', segmentated_image)
    encode_end = time.time()
    print("Color time %d ms" %((color_end - color_start)*1000))
    print("Encode time %d ms" %((encode_end - encode_start)*1000))
    # END OF REST API OUTPUT BOILERPLATE -------------------------
    #
    # PRINTS -----------------------------------------------------

    full_end = time.time()
    full_time = full_end - full_start
    avg_full_time = full_time / runTotal
    app.logger.info(' ')
    app.logger.info('\tProcessing Latency (data preparation + execution) :  \t%.2f ms (%.2f + %.2f)', (avg_full_time*1000), ((avg_full_time - avg_time_execution)*1000), int(avg_time_execution*1000))
    app.logger.info('\tTotal throughput (batch size) :                      \t%.2f fps (%d)', (runTotal/full_time), batch_size)
    # END OF PRINTS ----------------------------------------------
    # Return encoded image in string
    return seg_img_encoded

def preprocess_image(image, input_details_0):
   # Preprocess images
   # If models are int8, scaling is necessary
    image = image.astype(np.float32)
    image = image / NORM_FACTOR - 1.0
    if(input_details_0['dtype'] == np.uint8):
        input_scale, input_zero_point = input_details_0["quantization"]
        image = image / input_scale + input_zero_point
        image = tf.cast(x=image, dtype=tf.uint8)
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

@app.route('/api/infer',methods=['POST'])
def test():
    MODEL_PATH = "best_UNET_v3_INT8.tflite"
    BATCH_SIZE = 1
    cores = int(os.environ['CORES'])
    if(cores > 8 and cores < 0):
        NUM_THREADS = 4
    else:
        NUM_THREADS = cores
    r = request
    global initialized
    global warmed_up
    # Check if this is the first run
    if not initialized:
        print("init")
        app.logger.info("init")
        init_kernel(MODEL_PATH, BATCH_SIZE, NUM_THREADS) # This changes from case to case
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


app.run(host="0.0.0.0",port=3000) # This changes each time depending on the experiment

