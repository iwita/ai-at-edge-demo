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

IMAGE_PATH = "testing_0.png"
SEG_IMAGE_PATH = "out.png"
addr = 'http://localhost:3000'
test_url = addr + '/api/infer'

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
 
# prepare headers for http request
content_type = 'image/png'
headers = {'content-type': content_type}

img = cv2.imread(IMAGE_PATH ) # This needs to change
# encode image as jpeg
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
cv2.imwrite(SEG_IMAGE_PATH, seg_img)
# Uncomment to show images
# show_image(IMAGE_PATH, SEG_IMAGE_PATH)

