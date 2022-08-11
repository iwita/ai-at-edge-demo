#!/usr/bin/python3

import requests
import random
import time
import sys
import cv2
import json

addr = 'http://192.168.1.227:31334'
# addr = 'http://147.102.37.230:3000'
# addr = 'http://localhost:3000'
test_url = addr + '/api/infer'

# prepare headers for http request
content_type = 'image/jpeg'
# headers = {'content-type': content_type}

# img = cv2.imread('test_images/cat.jpg')
# encode image as jpeg
# _, img_encoded = cv2.imencode('.jpg', img)
# send http request with image and receive response
fileobj=open('./ImageNet_val_folder_1000_no_compression_images.zip', 'rb').read()
print(type(fileobj))
response = requests.post(test_url, data={"hi":"hello"}, files={"archive": ("images.zip", fileobj, "application/zip")})
# decode response
print(json.loads(response.text))
# response = requests.post('http://localhost:5000/schedule_v2', json={'app_name':'ada1234', 'app':app})