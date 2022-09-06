#!/usr/bin/python3

import requests
import random
import time
import sys
import cv2
import json
import zipfile
import datetime
IMAGE_ZIP_PATH = './ImageNet_val_folder_128_no_compression_images.zip' # Check the size 
DATASET_SIZE = 128
IMAGE_TO_SHOW = 8

addr = 'http://localhost:3001'
test_url = addr + '/api/infer'

fileobj=open(IMAGE_ZIP_PATH, 'rb').read()
print(type(fileobj))
start = datetime.datetime.now()
response = requests.post(test_url, data={"hi":"hello"}, files={"archive": ("images.zip", fileobj, "application/zip")})
end = datetime.datetime.now()
print("E2E Latency :\t %d ms" % (int((end-start).total_seconds()*1000)))
print("Throughput  :\t %d fps"  % (int(DATASET_SIZE/((end-start).total_seconds()))))
# decode response
with open('data.json', 'w') as f:
    json.dump(json.loads(response.text), f)
# Uncomment if you want "beautiful print"
# print(json.dumps(json.loads(response.text), indent=4))

# Uncomment if you want to show image
# show_image(IMAGE_TO_SHOW, IMAGE_ZIP_PATH)


def show_image(image_num, zip_path):
    # Requires cv2 installed on client
    zip_ref = zipfile.ZipFile(zip_path, 'r')
    zip_ref.extractall("./")
    zip_ref.close()
    # This is the name of the folder of the zip that contains all the images
    FOLDERNAME = "./ImageNet_val_folder"
    listimage=os.listdir(FOLDERNAME)
    listimage.sort()
    
    image = cv2.imread(os.path.join(FOLDERNAME,listimage[image_num]),cv2.IMREAD_COLOR)
    cv2.imshow('Classified Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return
