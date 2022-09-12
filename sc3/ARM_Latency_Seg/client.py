#!/usr/bin/python3

import requests
import random
import time
import sys
import cv2
import matplotlib.pyplot as plt
import json
import numpy as np
import datetime
import argparse

def get_seg_img_path(image_path):
    image_parts = image_path.split('/')
    image_folder = ""
    for i in range(len(image_parts) - 1):
        image_folder = image_folder + image_parts[i] + '/' 
    image_name = image_parts[-1].split('.')[0]
    image_type = "." + image_path.split('.')[-1]
    seg_image_path =image_folder + "seg_" + image_name + image_type
    return seg_image_path

def show_image(image_path, seg_image_path):
    image = plt.imread(image_path)
    seg_image = plt.imread(seg_image_path)
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(image)
    f.add_subplot(1,2, 2)
    plt.imshow(seg_image)
    plt.show(block=True)
    return

def app(image_path, address):

    seg_image_path = get_seg_img_path(image_path)
    test_url = address + '/api/infer'
    # prepare headers for http request
    content_type = 'image/png'
    headers = {'content-type': content_type}
    img = cv2.imread(image_path) # This needs to change
    _, img_encoded = cv2.imencode('.png', img)
    start = datetime.datetime.now()
    # send http request with image and receive response
    response = requests.post(test_url, data=img_encoded.tobytes(), headers=headers)
    end = datetime.datetime.now()
    print("E2E Latency :\t %.2f ms" % (((end-start).total_seconds()*1000)))
    print("Throughput  :\t %.2f fps"  % ((1.0/((end-start).total_seconds()))))
    # decode response
    seg_img=cv2.imdecode(np.frombuffer(response.content,np.uint8),cv2.IMREAD_COLOR)
    # save image
    cv2.imwrite(seg_image_path, seg_img)
    # Uncomment to show images
    # show_image(image_path, seg_image_path)

def main():
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image_path',  type=str, default="testing_0.png", help='Path of image to segmentate. Default is images/valid_0')
    ap.add_argument('-a', '--address',  type=str, default='http://localhost:3000', help='Address to connect to. Default is http://localhost:3000')

    args = ap.parse_args()

    print(' Command line options:')
    print ('--image_path        : ',args.image_path)
    print ('--address           : ',args.address)

    app(args.image_path, args.address)

if __name__ == '__main__':
    main()

