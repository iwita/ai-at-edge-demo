#!/usr/bin/python3

import requests
import random
import time
import sys
import cv2
import json
import datetime
IMAGE_ZIP_PATH = '../ImageNet_val_folder_1000_no_compression_images.zip'
DATASET_SIZE = 1000

addr = 'http://192.168.1.228:3002'
# addr = 'http://localhost:3000'
test_url = addr + '/api/infer'

fileobj=open(IMAGE_ZIP_PATH, 'rb').read()
print(type(fileobj))
start = datetime.datetime.now()
response = requests.post(test_url, data={"hi":"hello"}, files={"archive": ("images.zip", fileobj, "application/zip")})
end = datetime.datetime.now()
print("E2E Latency:\t %.2fms" % ((end-start).total_seconds()*1000))
print("Throughput:\t %.2fps"  % (DATASET_SIZE/((end-start).total_seconds())))
# decode response
print(json.loads(response.text))
# response = requests.post('http://localhost:5000/schedule_v2', json={'app_name':'ada1234', 'app':app})
