import requests
import random
import time
import sys
import cv2
import json


addr = 'http://192.168.1.228:31333'
#addr='http://192.168.1.228:3000'
test_url = addr + '/api/infer'


with open('./data/input.dat', 'rb') as f:
    r = requests.post(test_url, files={'file': f})

print(json.loads(r.text))
