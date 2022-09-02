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
#import vart
#import pathlib
#import xir
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

from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
#import torch

# Is the kernel already initialized?
initialized=False
warmed_up = False
logging.basicConfig(level=logging.DEBUG) #is this needed for logging?

# This variable is the model
sess = None
# This variable contains the json with the class names
CLASS_INDEX = None
CLASS_INDEX_path = 'imagenet_class_index.json' 
divider = '------------------------------------'

def init_kernel(model_path):
    st=datetime.now()
    global sess
    global CLASS_INDEX
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
    app.logger.info("{}".format(sess.get_providers()))
    app.logger.info("{}".format(sess.get_provider_options()))
    app.logger.info("{}".format(sess.get_session_options()))
    et = datetime.now()
    # Load CLASS_INDEX
    if(CLASS_INDEX is None):
        with open(CLASS_INDEX_path) as f:
            CLASS_INDEX = json.load(f)
    elapsed_time_i=et-st
    app.logger.info('Initialize time:\t' + str(elapsed_time_i.total_seconds()*1000) + 'ms')

def warm_up(batch_size):
    wm_start = datetime.now()
    global sess
    input_name = sess.get_inputs()[0].name
    preds = []
    x_dummy = tf.random.normal(shape =(batch_size, 224, 224, 3))
    x_input = tf.constant(x_dummy[:])
    output_data = sess.run([],{input_name: x_input.numpy()})[0]
    #torch.cuda.synchronize()
    for j in range(batch_size):
        probs = softmax(output_data[j].numpy())
        preds.append(decode_predictions(probs.reshape(1,-1), top=5)[0])
    for pred in preds:
        print(pred)
    wm_end = datetime.now()
    app.logger.info('Warmup time:\t' + str((wm_end-wm_start).total_seconds()*1000) + 'ms')

def inference(indata,batch_size, model_path):
    full_start = time.time()
    global sess
    input_name = sess.get_inputs()[0].name
    print(divider)
    app.logger.info("{:s}".format(divider))
    # REST API INPUT BOILERPLATE --------------------------------
    # We get the zip file with the images as Bytes
    zip_ref = zipfile.ZipFile(io.BytesIO(indata), 'r')
    zip_ref.extractall("./")
    zip_ref.close()
    # This is the name of the folder of the zip that contains all the images
    FOLDERNAME = "./ImageNet_val_folder_1000"
    listimage=os.listdir(FOLDERNAME)
    listimage.sort()
    runTotal = len(listimage)
    # END OF REST API INPUT BOILERPLATE -------------------------
    # 
    # CREATE AND PREPROCESS DATASET -----------------------------
    ds_val = tf.keras.preprocessing.image_dataset_from_directory(
        directory = FOLDERNAME,
        labels = None,
        label_mode = None,
        color_mode = "rgb",
        batch_size = batch_size,
        image_size = (224,224),
        shuffle = False
    )
    # Preprocess
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
    app.logger.info("Iterations: {}".format(iterations + remainder_iteration))
    for element in ds_val.take(iterations + remainder_iteration):
       x_test = element
       if(i == iterations): # Only last iteration
           x_input = tf.zeros(shape =(batch_size, 224, 224, 3))
           x_input_list = tf.unstack(x_input)
           x_input_list[0:x_test.shape[0]] = x_test[:]
           x_input = tf.stack(x_input_list)
       else:
           x_input =tf.constant(x_test[:])
       start = time.time()
       # Inference
       output_data = sess.run([],{input_name: x_input.numpy()})[0]
       # This is really important. GPU inference run asynchronously, we need to wait for the process to end before using time()
       # Tensorflow doesn't have this fucntionality, therefore torch.cuda.synchronize is used
       #torch.cuda.synchronize()
       end = time.time()
       timetotal_execution += end - start # Time in seconds, e.g. 5.38091952400282 
       if(i == iterations): # Only last iteration
           valid_outputs = x_test.shape[0]
       else:
           valid_outputs = batch_size
       # Calculate the probs
       for j in range(valid_outputs):
           probs = softmax(output_data[j].numpy())
           preds.append(decode_predictions(probs.reshape(1,-1), top=5)[0])
           #preds.append(tf.keras.applications.imagenet_utils.decode_predictions(probs.reshape(1,-1), top=5)[0])
       i = i + 1
    # END OF EXPERIMENT ------------------------------------------
    #
    # BENCHMARKS -------------------------------------------------
    avg_time_execution = timetotal_execution / (iterations + remainder_iteration)
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
    full_end = time.time()
    full_time = full_end - full_start
    avg_full_time = full_time/ (iterations + remainder_iteration)
    IMAGE_TO_SHOW = 796
    to_print = out_dict[listimage[IMAGE_TO_SHOW]]
    app.logger.info(' ')
    app.logger.info('Processing Latency: (data preparation + execution) :\t%.2fms (%.2f + %.2f)', avg_full_time*1000, (avg_full_time - avg_time_execution)*1000, avg_time_execution*1000)
    app.logger.info('Total throughput (batch_size) in frames per second :\t\t%.2ffps (%d)', runTotal/full_time, batch_size)
    app.logger.info(' ')
    app.logger.info('AIF output: \t class = %s (%03d: %.2f, %03d: %.2f, %03d: %.2f, %03d: %.2f, %03d: %.2f)',
                    to_print[0][1], to_print[0][0], to_print[0][2], to_print[1][0], to_print[1][2],
                    to_print[2][0], to_print[2][2], to_print[3][0], to_print[3][2], to_print[4][0], to_print[4][2])
    app.logger.info(' ')
    app.logger.info('AIF output, ImageName class=name (top 5 classes with percentages):')
    app.logger.info('Image: %s class = %s (%03d: %.2f, %03d: %.2f, %03d: %.2f, %03d: %.2f, %03d: %.2f)', listimage[IMAGE_TO_SHOW],
                    to_print[0][1], to_print[0][0], to_print[0][2], to_print[1][0], to_print[1][2],
                    to_print[2][0], to_print[2][2], to_print[3][0], to_print[3][2], to_print[4][0], to_print[4][2])

    # END OF PRINTS ----------------------------------------------
    # Return Dictionary
    return out_dict

def preprocess(x_test, MODEL_PATH):
   # Preprocess images
   # if "ResNet50" in MODEL_PATH:
   x_test = tf.keras.applications.resnet50.preprocess_input(x_test)
   return x_test

def softmax(logits):
   # Works for 1D np arrays
   scores = logits - np.max(logits)
   probs = np.exp(scores.astype(np.float64))/np.sum(np.exp(scores.astype(np.float64)))
   return probs

def decode_predictions(preds, top=5):
    # tf.keras.applications.imagenet_utils.decode_predictions
    # CLASS_INDEX must come from imagenet_class_index.json
    global CLASS_INDEX
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(int(i), CLASS_INDEX[str(i)][1], pred[i]) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results

app=Flask(__name__)

@app.route('/api/infer',methods=['POST'])
def test():
    MODEL_PATH = "ResNet50_ImageNet_70_90_7_76GF_B32.onnx"
    BATCH_SIZE = 32
    r = request
    global initialized
    global warmed_up
    # Check if this is the first run
    if not initialized:
        print("init")
        app.logger.info("init")
        init_kernel(MODEL_PATH) # This changes from case to case
        initialized=True
    if not warmed_up:
        print("warm_up")
        app.logger.info("warm_up")
        warm_up(BATCH_SIZE)
        warmed_up=True
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

