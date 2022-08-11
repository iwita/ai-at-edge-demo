#!/usr/bin/python3

import requests
import random
import time
import sys
import cv2
import json
import numpy as np

SEG_IMAGE_PATH = "../seg_img_test/img_test/testing_0.png"

addr = 'http://192.168.1.228:3000'

test_url = addr + '/api/infer'

# prepare headers for http request
content_type = 'image/png'
headers = {'content-type': content_type}

img = cv2.imread(SEG_IMAGE_PATH ) # This needs to change
# encode image as jpeg
_, img_encoded = cv2.imencode('.png', img)
# send http request with image and receive response
response = requests.post(test_url, data=img_encoded.tobytes(), headers=headers)
# decode response
# print(response.text.)
print(response.content)
seg_img=cv2.imdecode(np.frombuffer(response.content,np.uint8),cv2.IMREAD_COLOR)
print(seg_img)
cv2.imwrite("./out.png", seg_img)