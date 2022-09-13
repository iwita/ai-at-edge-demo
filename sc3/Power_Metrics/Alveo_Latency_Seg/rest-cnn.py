#!/usr/bin/python3

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

# Initialize
all_dpu_runners = []

# Is the kernel already initialized?
initialized=False

#logging.basicConfig(level=logging.DEBUG)
# app.logger.basicConfig(level=app.logger.DEBUG) #is this needed for app.logger?

# This variable stores the inferred label(s) of the incoming images
out_q = None
from power_monitor_thread import Alveo_Power_Monitor_Thread
divider = '------------------------------------'

def init_kernel(threads,model):
    st=datetime.now()
    global all_dpu_runners
    #Load Model
    all_dpu_runners = []    # Because we re-initialize, empty the list
    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)
    #Create the DPU runners
    for i in range(threads):
        try:
            all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))
        except Exception as e:
            print(e)
    et = datetime.now()
    elapsed_time_i=et-st
    logging.info('Initialize time :\t' + str(int(elapsed_time_i.total_seconds()*1000)) + ' ms')

def inference(image_path,batch_size, threads, num_of_runs):
    full_start = time.time()
    global all_dpu_runners
    global out_q
    print(divider)
    # REST API INPUT BOILERPLATE --------------------------------
    # Data --> Image . Assume we get the data in numpyarray of image encoded bytes
    rest_api_input_start = time.time()
    img = cv2.imread(image_path)
    runTotal = 1
    out_q = [None] * runTotal
    rest_api_output_end = time.time() 
    print("Rest API Input %d ms" %((rest_api_output_end - rest_api_input_start)*1000))
    # END OF REST API INPUT BOILERPLATE -------------------------
    # 
    # CREATE AND PREPROCESS DATASET -----------------------------
    # input scaling
    pre_start = time.time()
    input_fixpos = all_dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2**input_fixpos
    # Input images
    image_list = []
    # Preprocess
    for i in range(runTotal):
        image_list.append(preprocess_image(img, input_scale))
    pre_end= time.time()
    print("Preprocess time %d ms" %((pre_end - pre_start)*1000))
    # END OF CREATE AND PREPROCESS DATASET ----------------------
    #
    # EXPERIMENT ------------------------------------------------
    # '''run threads '''
    thread_creation_start = time.time()
    logging.info("-------------Experiment-------------")
    logging.info('Starting {:d} threads...'.format(runTotal))
    threadAll = []
    start=0
    #for i in range(threads):
    #    if (i==threads-1):
    #        end = len(image_list)
    #    else:
    #        end = start+(len(image_list)//threads)
    #    in_q = image_list[start:end]
    #    #Create the thread
    #    t1 = threading.Thread(target=runDPU, args=(i,start,all_dpu_runners[i], in_q, batch_size))
    #    threadAll.append(t1)
    #    start=end
    #t1 = threading.Thread(target=runDPU, args=(0, 0, all_dpu_runners[0], image_list, 1))
    thread_creation_end = time.time()
    print("Thread creation time %d ms" %((thread_creation_end - thread_creation_start)*1000))
    time1 = time.time()
    t2 = Alveo_Power_Monitor_Thread(sleep_time = 0.1, device_id='0000:00:08.0')
    t2.start()
    for i in range(num_of_runs):
        runDPU(0, 0, all_dpu_runners[0], image_list, 1)
    #Run the Thread
    t2.terminate()
    t2.join()
    power_metrics = t2.get_results()
    #t1.start()
    #t1.join()
    #for x in threadAll:
    #    x.start()
    #for x in threadAll:
    #    x.join()
    time2 = time.time()
    post_start = time.time()
    y_pred1_i = np.asarray(out_q) # Expected shape is (1, HEIGHT, WIDTH) and each index is the number of the color
    post_end = time.time()
    print("Postprocess time %d ms" %((post_end - post_start)*1000))
    # END OF EXPERIMENT ------------------------------------------
    #
    # BENCHMARKS -------------------------------------------------
    timetotal_execution = time2 - time1
    avg_time_execution = timetotal_execution/runTotal
    # END OF BENCHMARKS ------------------------------------------
    #
    # REST API OUTPUT BOILERPLATE --------------------------------
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

def runDPU(id,start,dpu,img,batch):
    '''get tensor'''
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)

    # we can avoid output scaling if use argmax instead of softmax
    #output_fixpos = outputTensors[0].get_attr("fix_point")
    #output_scale = 1 / (2**output_fixpos)
    #batchSize = input_ndim[0]
    if(batch == 0):
        batchSize = input_ndim[0]  
    elif(batch > input_ndim[0]):
        batchSize = input_ndim[0]
        logging.info("Batch {} was bigger than max {}. Resized to {}".format(batch, input_ndim[0], input_ndim[0]))
    elif(batch < 0):
        batchSize = input_ndim[0]
        logging.info("Invalid size for batch {}. Resized to {}".format(batch, input_ndim[0]))
    elif(batch > 0 and batch <= input_ndim[0]):
        batchSize = batch
    else:
        logging.info("Unexpected Error")
    # print("output_ndim %d" %(output_ndim[0]))
    n_of_images = len(img)
    count = 0
    write_index = start
    outputData = []
    time_total = 0.0
    for i in range(n_of_images):
        outputData.append([np.empty(output_ndim, dtype=np.int8, order="C")])
    while count < n_of_images:
        if (count+batchSize<=n_of_images):
            runSize = batchSize
        else:
            runSize=n_of_images-count

        '''prepare batch input/output '''
        inputData = []
        inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]
        #inputData[0] = img[count:count+runSize]
        imageRun = inputData[0]
        imageRun[0:runSize] = img[count:count+runSize]
        '''run with batch '''
        job_id = dpu.execute_async(inputData,outputData[count])
        dpu.wait(job_id)
        for j in range(runSize):
            out_q[start+count+j] = np.argmax(outputData[count][0][j],axis=2)
        count = count + runSize
    return

def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]

def preprocess_image(image, fix_scale):
    #Image normalization
    #Args:     Image and label
    #Returns:  normalized image and unchanged labels
    image= image.astype(np.float32)
    image = image / NORM_FACTOR - 1.0
    image = image * fix_scale
    image = image.astype(np.int8)
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
    MODEL_PATH = 'customcnn.xmodel'
    THREADS=1
    BATCH_SIZE=1
    r = request
    global initialized
    # Check if this is the first run
    if not initialized:
        init_kernel(THREADS,MODEL_PATH)
        print("init")
        logging.info("init")
        initialized=True
    # Call the service here
    print("inference")
    logging.info("inference")
    seg_img_string = inference(image_path, BATCH_SIZE, THREADS, num_of_runs)

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
    logging.basicConfig(filename='logs/sc3_alveo_power_metrics_{}.log'.format(args.num_of_runs), level=logging.INFO)

    logging.info(' Command line options:')
    logging.info('--image_path       : {}'.format(args.image_path))
    logging.info('--num_of_runs      : {}'.format(args.num_of_runs))

    test(args.image_path, args.num_of_runs)

if __name__ == '__main__':
    main()
