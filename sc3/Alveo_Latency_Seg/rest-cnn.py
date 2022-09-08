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
THREADS=1
BATCH=1

# Is the kernel already initialized?
initialized=False

logging.basicConfig(level=logging.DEBUG)
# app.logger.basicConfig(level=app.logger.DEBUG) #is this needed for app.logger?

# This variable stores the inferred label(s) of the incoming images
out_q = None
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
    elasped_time_i=et-st
    app.logger.info('Initialize time :\t' + str(int(elapsed_time_i.total_seconds()*1000)) + ' ms')

def inference(indata,threads):
    full_start = time.time()
    global all_dpu_runners
    global out_q
    print(divider)
    # REST API INPUT BOILERPLATE --------------------------------
    # Data --> Image . Assume we get the data in numpyarray of image encoded bytes
    img=cv2.imdecode(np.fromstring(indata.data,np.uint8),cv2.IMREAD_COLOR)
    runTotal = 1
    out_q = [None] * runTotal
    # END OF REST API INPUT BOILERPLATE -------------------------
    # 
    # CREATE AND PREPROCESS DATASET -----------------------------
    # input scaling
    input_fixpos = all_dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2**input_fixpos
    # Input images
    image_list = []
    # Preprocess
    for i in range(runTotal):
        image_list.append(preprocess_image(img, input_scale))
    # END OF CREATE AND PREPROCESS DATASET ----------------------
    #
    # EXPERIMENT ------------------------------------------------
    # '''run threads '''
    app.logger.info("-------------Experiment-------------")
    app.logger.info('Starting {:d} threads...'.format(runTotal))

    print(image_list[0])
    print(image_list[0].shape)

    threadAll = []
    start=0
    for i in range(threads):
        if (i==threads-1):
            end = len(image_list)
        else:
            end = start+(len(image_list)//threads)
        in_q = image_list[start:end]
        #Create the thread
        t1 = threading.Thread(target=runDPU, args=(i,start,all_dpu_runners[i], in_q, BATCH))
        threadAll.append(t1)
        start=end
    time1 = time.time()
    #Run the Thread
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    time2 = time.time()
    y_pred1_i = np.asarray(out_q) # Expected shape is (1, HEIGHT, WIDTH) and each index is the number of the color
    # END OF EXPERIMENT ------------------------------------------
    #
    # BENCHMARKS -------------------------------------------------
    timetotal_execution = time2 - time1
    avg_time_execution = timetotal_execution/runTotal
    #fps = float(runTotal / timetotal)
    #print(divider)
    #app.logger.info("{:s}".format(divider))
    #print("Throughput=%.2f fps, total frames = %d, time=%.4f seconds" %(fps, runTotal, timetotal))
    #app.logger.info("Throughput=%.2f fps, total frames = %d, time=%.4f seconds" %(fps, runTotal, timetotal))
    # END OF BENCHMARKS ------------------------------------------
    #
    # REST API OUTPUT BOILERPLATE --------------------------------
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
    app.logger.info('\tProcessing Latency (data preparation + execution) :  \t%.2f ms (%.2f + %.2f)', (avg_full_time*1000), ((avg_full_time - avg_time_execution)*1000), int(avg_time_execution*1000))
    app.logger.info('\tTotal throughput (batch size) :                      \t%.2f fps (%d)', (runTotal/full_time), batch_size)
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
    app.logger.info("input_ndim: {}".format(input_ndim))
    app.logger.info("output_ndim: {}".format(output_ndim))
    #batchSize = input_ndim[0]
    if(batch == 0):
        batchSize = input_ndim[0]  
    elif(batch > input_ndim[0]):
        batchSize = input_ndim[0]
        app.logger.info("Batch {} was bigger than max {}. Resized to {}".format(batch, input_ndim[0], input_ndim[0]))
    elif(batch < 0):
        batchSize = input_ndim[0]
        app.logger.info("Invalid size for batch {}. Resized to {}".format(batch, input_ndim[0]))
    elif(batch > 0 and batch <= input_ndim[0]):
        batchSize = batch
    else:
        app.logger.info("Unexpected Error")
    app.logger.info("batchSize %d" %(batchSize))
    app.logger.info("output_ndim %d" %(output_ndim[0]))
    # print("output_ndim %d" %(output_ndim[0]))
    n_of_images = len(img)
    app.logger.info("n_of_images %d" %(n_of_images))
    count = 0
    write_index = start
    app.logger.info("write_index %d" %(start))
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

def denormalize_image(image):
    return (image+1.0)*NORM_FACTOR/255.0

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
    r = request
    global initialized
    # Check if this is the first run
    if not initialized:
        init_kernel(THREADS,'customcnn.xmodel')
        print("init")
        app.logger.info("init")
        initialized=True
    # Call the service here
    print("inference")
    app.logger.info("inference")
    seg_img_string = inference(r, THREADS)
    #seg_img_string = inference(np.fromstring(r.data,np.uint8), THREADS)
    # Returns np.array as string. Need to imdecode in client
    return Response(response=seg_img_string.tobytes(),status=200,mimetype="image/png") 

@app.route('/api/test2',methods=['POST'])
def test2():

    return Response(response=json.dumps({"res":"ok"}),status=200,mimetype="application/json")


app.run(host="0.0.0.0",port=3000)
