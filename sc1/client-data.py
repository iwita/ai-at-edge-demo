#!/usr/bin/python3

import requests
import random
import sys
import cv2
import json
import datetime
import time
import os
import numpy

addr = 'http://192.168.1.228:31333'
# addr='http://0.0.0.0:5000'
test_url = addr + '/api/infer'
# test_url = addr + '/api/test2'


# with open('./data/input.dat', 'rb') as f:
    
#     r = requests.post(test_url, files={'file': f})
   
# end = datetime.datetime.now()
# print((end-start))


size=50
fl_name='./data/input.dat'

V = numpy.empty(size, float)

with open(fl_name, "r") as fl:
    V = [float(next(fl)) for x in range(size)]
# print(V)
# print(str(V))
data ={"array": V}
start = datetime.datetime.now()
r = requests.post(test_url, json=data)
end = datetime.datetime.now()
print("E2E Latency:\t %.2fms" % ((end-start).total_seconds()*1000))
print("Throughput:\t %.2frps" % (1000/((end-start).total_seconds()*1000)))

# encoded = json.dumps(V, cls=NumbpyArrayEncoder)
# print(encoded)

# with open('./data/input.dat', 'rb') as f:   
#     time.sleep(2)
#     # with open('./data/input.dat', 'rb') as f:
#     start = datetime.datetime.now()
#     r = requests.post(test_url, files={'file': f})
#     end = datetime.datetime.now()
#     print((end-start))
#     print(json.loads(r.text))

# time.sleep(2)

# print((end-start))
print(json.loads(r.text))
