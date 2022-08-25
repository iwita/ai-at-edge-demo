import requests
import random
import ctypes
import os
import cv2
import json
import numpy 
import linecache


addr = 'http://192.168.1.228:31333'
#addr='http://192.168.1.228:3000'
test_url = addr + '/api/infer'
size=50
fl_name='./data/input.dat'

V = numpy.empty(size, float)

with open(fl_name, "r") as fl:
    V = [float(next(fl)) for x in range(size)]

# with open('./data/input.dat', 'rb') as f:
r = requests.post(test_url, files={'file': f})

print(json.loads(r.text))
