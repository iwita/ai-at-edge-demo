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

# Initialize
all_dpu_runners = []
THREADS=3
BATCH=0 # This means that automatically go to biggest batch (3)

# Is the kernel already initialized?
initialized=False

logging.basicConfig(level=logging.DEBUG)


# This variable stores the inferred label(s) of the incoming images
out_q = None

MODEL = "ResNet50"

divider = '------------------------------------'

def init_kernel(threads,model):
    st=datetime.now()
    # global out_q
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
    # app.logger.info("Initialization time:" + str(elasped_time_i.total_seconds()*1000)+ "ms")
    app.logger.info('Initialization time: %.2fms', elasped_time_i.total_seconds()*1000)

def inference_local(indata,threads):
    # global all_dpu_runners
    # global out_q
    # input scaling
    # input_fixpos = all_dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
    # input_scale = 2**input_fixpos
    # print(divider)
    # If we want to get multiple images, we should pass them in a python list
    # Data --> Image . Assume we get the data in numpyarray of image encoded bytes
    zip_ref = zipfile.ZipFile(io.BytesIO(indata), 'r')
    # zip_ref=indata


    # dataset_dir = "Images"
    zip_ref.extractall("./")
    zip_ref.close()
    listimage=os.listdir("./ImageNet_val_folder_1000")
    
    runTotal = len(listimage)
    print(listimage, runTotal)
    # Append pre-processed images to img list
    # Let's keep it as is for now (to allow for bath processing in our tests)
    # for i in range(runTotal):

    # We could support batch mode later on
    # runTotal=1
    # data = cv2.imread(img)
    # print(img)
    # print(type(img))
    # print(img.shape)
    # proc_img = preprocess_image(img, input_scale)
    # print("Processed Image")
    # print(proc_img)
    # print(proc_img.shape)
    # image_list.append(preprocess_image(img, input_scale))
    # print(image_list[0])

def inference(indata,threads):

    global all_dpu_runners
    global out_q
    # input scaling
    input_fixpos = all_dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2**input_fixpos
    print(divider)
    # If we want to get multiple images, we should pass them in a python list
    # Data --> Image . Assume we get the data in numpyarray of image encoded bytes
    zip_ref = zipfile.ZipFile(io.BytesIO(indata), 'r')
    # dataset_dir = "Images"
    zip_ref.extractall("./")
    zip_ref.close()
    FOLDERNAME = "./ImageNet_val_folder_1000"
    listimage=os.listdir(FOLDERNAME)
    
    runTotal = len(listimage)
    out_q = [None] * runTotal
    # print('Pre-processing',runTotal,'images...')
    
    # Input images
    image_list = []

    # data = cv2.imread(img)
    for i in range(runTotal):
        image_list.append(preprocess_image(cv2.imread(os.path.join(FOLDERNAME,listimage[i]),cv2.IMREAD_COLOR), input_scale))
    # '''run threads '''
    print('Starting',threads,'threads...')
    threadAll = []
    start=0
    my_time = [None] * threads #Trick to get the time later
    for i in range(threads):
        if (i==threads-1):
            end = len(image_list)
        else:
            end = start+(len(image_list)//threads)
        in_q = image_list[start:end]
        #Create the thread
        t1 = threading.Thread(target=runDPU, args=(i,start,all_dpu_runners[i], in_q, my_time, BATCH))
        threadAll.append(t1)
        start=end

    time1 = datetime.now()
    #Run the Thread
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    time2 = datetime.now()
    timetotal = time2 - time1
    timetotal_v2 =  max(my_time)
    print("timetotal %4f , time_total_v2 %.4f" %(timetotal.total_seconds(), timetotal_v2))
    # We prefer timetotal, seems more reliable 
    fps = float(runTotal / timetotal.total_seconds())
    print(divider)
    print("Throughput=%.2f fps, total frames = %d, time=%.4f seconds" %(fps, runTotal, timetotal.total_seconds()))
    print(len(out_q))
    preds = []
    for i in range(len(out_q)):
        #print(out_q[i].shape)
        probs = softmax(out_q[i])
        #print(probs)
        preds.append(tf.keras.applications.imagenet_utils.decode_predictions(probs[0], top=5)[0])
    
    #Write results on txt and return them??
    # with open("predictions.txt", "w") as myfile:
    #    for i in range(len(preds)):
    #        myfile.write("Image {:03d}:\n {} \n". format(i, preds[i]))
    
    #for i in range(len(preds)):
    #    print("Image {:s}:\n {} ". format(listimage[i], preds[i]))
    
    out_dict = {}
    for i in range(len(preds)):
        out_dict[listimage[i]] = preds[i]
    return out_dict
    # return

def runDPU(id,start,dpu,img,my_time,batch):
    '''get tensor'''
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)

    # we can avoid output scaling if use argmax instead of softmax
    #output_fixpos = outputTensors[0].get_attr("fix_point")
    #output_scale = 1 / (2**output_fixpos)
    logging.info("input_ndim: {}".format(input_ndim))
    logging.info("output_ndim: {}".format(output_ndim))
    batchSize = input_ndim[0]
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
    logging.info("batchSize %d" %(batchSize))
    logging.info("batchSize %d" %(batchSize))
    logging.info("output_ndim %d" %(output_ndim[0]))
    n_of_images = len(img)
    logging.info("n_of_images %d" %(n_of_images))
    count = 0
    write_index = start
    logging.info("write_index %d" %(start))
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
        time1 = time.time()
        imageRun = inputData[0]
        imageRun[0:runSize] = img[count:count+runSize]
        '''run with batch '''
        job_id = dpu.execute_async(inputData,outputData[count])
        dpu.wait(job_id)
        time2 = time.time()
        for j in range(runSize):
            #out_q[start+count+j] = np.argmax(outputData[count][0][j])
            out_q[start+count+j] = outputData[count][0][j]
        time_total += time2 - time1
        count = count + runSize 
    my_time[id] = time_total
    logging.info("id: {} time: {:.3f}".format(id, time_total))

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
    #Returns:  normalized image and unchanged label
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #CV2 Loads on BGR, but we need RGB 
    if(MODEL == "ResNet50"):
        image = tf_preprocess_input(image, "caffe")
        image = cv2.resize(image, dsize=(224,224), interpolation=cv2.INTER_CUBIC)
	#image = tf.keras.applications.resnet50.preprocess_input(image)
    elif(MODEL == "MobileNetV1"):
        image = image.astype(np.float32, copy=False)/127.5 - 1.0
        image = cv2.resize(image, dsize=(224,224), interpolation=cv2.INTER_CUBIC)
    elif(MODEL == "LeNet5"):
        image = image.astype(np.float32, copy=False)/255.0
        image = cv2.resize(image, dsize=(32,32), interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image.reshape(32, 32, 1)
    	#image = image.astype(np.float32, copy=False)/255.0
    elif(MODEL == "ResNet152" or MODEL == "InceptionV4"):
        image = image.astype(np.float32, copy=False)/127.5 - 1.0
        image = cv2.resize(image, dsize=(299,299), interpolation=cv2.INTER_CUBIC)
    image = image * fix_scale
    image = image.astype(np.int8)
    return image

def tf_preprocess_input(x,mode):
    if not issubclass(x.dtype.type, np.floating):
        x = x.astype(np.float32, copy=False)
    if mode == 'tf':
        x /= 127.5
        x -= 1.
        return x
    elif mode == 'torch':
        x /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
      # 'RGB'->'BGR'
        x = x[..., ::-1]
        mean = [103.939, 116.779, 123.68]
        std = None
  # Zero-center by mean pixel
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    if std is not None:
        x[..., 0] /= std[0]
        x[..., 1] /= std[1]
        x[..., 2] /= std[2]
    return x

def softmax(logits):
   # Works for 1D np arrays
   scores = logits - np.max(logits)
   probs = np.exp(scores.astype(np.float64))/np.sum(np.exp(scores.astype(np.float64)))
   return probs

# def main():
#     print("hello")


app=Flask(__name__)



@app.route('/api/infer',methods=['POST'])
def test():
    r = request
    global initialized
    # global initialized
    # Check if this is the first run
    if not initialized:
        print(THREADS)
        init_kernel(THREADS,'customcnn.xmodel')
        print("init")
        initialized=True
        # app.logger.info(("init_kernel")
        # init_kernel()

    # print(r.get_json())
    print(r.files['archive'])
    file = r.files['archive']
    # print(request.data)
    # print(r)
    # file = r.files
    # print(type(r.data))
    # print(type(r.files))
    # data=r.data
    # print(data.archive)
    
    # print(file)
    file_like_object = file.read()
    


    print(type(file))
    print(type(file_like_object))

    # nparr=np.fromstring(r.data,np.uint8)
    # print(nparr)
    # print(type(nparr))
    # print(nparr.shape)

    # Call the service here
    print("inference")
    # inference_local(file_like_object,THREADS)
    preds = inference(file_like_object, THREADS)


    # img=cv2.imdecode(nparr,cv2.IMREAD_COLOR)

    # f = open('output'+str(counter_id)+'.jpg',"wb")
    # f.write(img)
    # print("Image Received")
    # f.close


    return Response(response=json.dumps(str(preds)),status=200,mimetype="application/json")

@app.route('/api/test2',methods=['POST'])
def test2():

    return Response(response=json.dumps({"res":"ok"}),status=200,mimetype="application/json")


app.run(host="0.0.0.0",port=3000)
