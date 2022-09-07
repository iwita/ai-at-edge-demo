#!/usr/bin/python3

import requests
import random
import time
import sys
import json
import datetime
IMAGE_ZIP_PATH = './ImageNet_val_folder_32_no_compression_images.zip' # Check the size 
DATASET_SIZE = 32
IMAGE_TO_SHOW = 0
addr = 'http://localhost:3001'
test_url = addr + '/api/infer'

def print_AIF_output():
    with open('data.json') as json_file:
        data = json.load(json_file)
        to_print_0 = data['000_000.jpg']
        to_print_1 = data['001_000.jpg']
        print('AIF output: Image_1 class name (top-5 classes prob.) \tclass = "%s" ("%03d": %.2f, "%03d": %.2f, "%03d": %.2f, "%03d": %.2f, "%03d": %.2f)',
                    to_print_0[0][1], to_print_0[0][0], to_print_0[0][2], to_print_0[1][0], to_print_0[1][2],
                    to_print_0[2][0], to_print_0[2][2], to_print_0[3][0], to_print_0[3][2], to_print_0[4][0], to_print_0[4][2])
        print('            Image_2 class name (top-5 classes prob.) \tclass = "%s" ("%03d": %.2f, "%03d": %.2f, "%03d": %.2f, "%03d": %.2f, "%03d": %.2f)',
                    to_print_1[0][1], to_print_1[0][0], to_print_1[0][2], to_print_1[1][0], to_print_1[1][2],
                    to_print_1[2][0], to_print_1[2][2], to_print_1[3][0], to_print_1[3][2], to_print_1[4][0], to_print_1[4][2])
        print('            ...')
    return

def show_image(image_num, zip_path):
    # Requires cv2 installed on client
    import cv2
    import zipfile
    import shutil
    # This is the name of the folder of the zip that contains all the images
    FOLDERNAME = "./ImageNet_val_folder"
    shutil.rmtree(FOLDERNAME)
    zip_ref = zipfile.ZipFile(zip_path, 'r')
    zip_ref.extractall("./")
    zip_ref.close()
    listimage=os.listdir(FOLDERNAME)
    listimage.sort()

    runTotal = len(listimage)
    if(image_num >= runTotal):
         image_num = 0

    image = cv2.imread(os.path.join(FOLDERNAME,listimage[image_num]),cv2.IMREAD_COLOR)
    cv2.imshow('Classified Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

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
# Uncomment if you want "beautiful print"nano 
# print(json.dumps(json.loads(response.text), indent=4))

print_AIF_output()

# Uncomment if you want to show image
# show_image(IMAGE_TO_SHOW, IMAGE_ZIP_PATH)
