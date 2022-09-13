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
import onnxruntime as ort
import onnx
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
#logging.basicConfig(level=logging.DEBUG) #is this needed for logging?

# This variable is the model
sess = None
from power_monitor_thread import AGX_Power_Monitor_Thread
divider = '------------------------------------'

def init_kernel(model_path,batch_size):
    st=datetime.now()
    global sess
    providers = [
         ('TensorrtExecutionProvider',{
          'device_id' : 0,
          'trt_fp16_enable' : False,
          'trt_int8_enable' : True,
          'trt_int8_calibration_table_name' : 'calibration.flatbuffers',
          'trt_engine_cache_enable' : False
         }),
         ('CUDAExecutionProvider', {
          'device_id' : 0,
         })
    ] 
    #Load Model
    sess_opt = ort.SessionOptions()
    sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(path_or_bytes=model_path, sess_options=sess_opt, providers=providers)
    # Useful for Debugging
    logging.info("{}".format(sess.get_providers()))
    logging.info("{}".format(sess.get_provider_options()))
    logging.info("{}".format(sess.get_session_options()))
    et = datetime.now()
    elapsed_time_i=et-st
    logging.info('Initialize time :\t' + str(int(elapsed_time_i.total_seconds()*1000)) + ' ms')

def warm_up(batch_size):
    wm_start = datetime.now()
    global sess
    input_name = sess.get_inputs()[0].name
    x_dummy = np.zeros(shape=(batch_size,224,224,3), dtype=np.float32)
    #x_dummy = tf.random.normal(shape =(batch_size, 224, 224, 3))
    #x_input = tf.constant(x_dummy[tf.newaxis,0,:])
    #output_data = sess.run([],{input_name: x_input.numpy()})[0]
    output_data = sess.run([],{input_name: x_dummy})[0]
    #torch.cuda.synchronize()
    wm_end = datetime.now()
    logging.info('Warmup time :\t' + str(int((wm_end-wm_start).total_seconds()*1000)) + ' ms')

def inference(image_path,batch_size, model_path, num_of_runs):
    full_start = time.time()
    global sess
    input_name = sess.get_inputs()[0].name
    # REST API INPUT BOILERPLATE --------------------------------
    # Data --> Image . Assume we get the data in numpyarray of image encoded bytes
    rest_api_input_start = time.time()
    img = cv2.imread(image_path)
    runTotal = 1
    rest_api_output_end = time.time() 
    print("Rest API Input %d ms" %((rest_api_output_end - rest_api_input_start)*1000))
    # END OF REST API INPUT BOILERPLATE -------------------------
    # 
    # CREATE AND PREPROCESS DATASET -----------------------------
    pre_start = time.time()
    x_test = preprocess_image(img)
    pre_end= time.time()
    print("Preprocess time %d ms" %((pre_end - pre_start)*1000))
    # END OF CREATE AND PREPROCESS DATASET ----------------------
    #
    # EXPERIMENT ------------------------------------------------
    logging.info("Dataset size: {}".format(runTotal))
    #num_samples = runTotal
    #x_input = tf.constant(x_test[tf.newaxis,:])
    t1 = AGX_Power_Monitor_Thread(sleep_time = 0.1)
    start =  time.time()
    t1.start()
    for i in range(num_of_runs):
        output_data = sess.run([],{input_name: x_test[np.newaxis,:]})[0]
    #output_data = sess.run([],{input_name: x_input.numpy()})[0]
    #torch.cuda.synchronize()
    t1.terminate()
    t1.join()
    power_metrics = t1.get_results()
    end = time.time()
    post_start = time.time()
    y_pred1_i = np.argmax(output_data, axis=3) # Expected shape is (1, HEIGHT, WIDTH) and each index is the number of the color??
    post_end = time.time()
    print("Postprocess time %d ms" %((post_end - post_start)*1000))
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
    logging.info(' ')
    logging.info('\tProcessing Latency (data preparation + execution) :  \t%.2f ms (%.2f + %.2f)', (avg_full_time*1000), ((avg_full_time - avg_time_execution)*1000), (avg_time_execution*1000))
    logging.info('\tTotal throughput (batch size) :                      \t%.2f fps (%d)', (runTotal/full_time), batch_size)
    logging.info("Num_of_queries {:d}, Avg Power {:.2f} W, Total Energy {:.2f} J".format(power_metrics[0], power_metrics[1], power_metrics[1] * (avg_time_execution)))
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


def test(image_path, num_of_runs):
    MODEL_PATH = "best_UNET_v3_B1.onnx"
    BATCH_SIZE = 1
    r = request
    global initialized
    global warmed_up
    # Check if this is the first run
    if not initialized:
        print("init")
        logging.info("init")
        init_kernel(MODEL_PATH, BATCH_SIZE) # This changes from case to case
        initialized=True
    if not warmed_up:
        print("warm_up")
        logging.info("warm_up")
        warm_up(BATCH_SIZE)
        warmed_up=True
    # Call the service here
    print("inference")
    logging.info("inference")
    seg_img_string = inference(image_path, BATCH_SIZE, MODEL_PATH, num_of_runs)

def main():
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image_path',  type=str, default="testing_0.png", help='Path of image to segmentate. Default is images/valid_0')
    ap.add_argument('-n', '--num_of_runs',  type=int, default=10000, help='How many runs')
    args = ap.parse_args()

    print(' Command line options:')
    print ('--image_path       : ',args.image_path)
    print ('--num_of_runs      : ',str(args.num_of_runs))

    print(divider)
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(filename='logs/sc3_agx_power_metrics_{}.log'.format(args.num_of_runs), level=logging.INFO)

    logging.info(' Command line options:')
    logging.info('--image_path       : {}'.format(args.image_path))
    logging.info('--num_of_runs      : {}'.format(args.num_of_runs))

    test(args.image_path, args.num_of_runs)

if __name__ == '__main__':
    main()

