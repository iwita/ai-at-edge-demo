FROM tensorflow/tensorflow:2.8.0-gpu

RUN /usr/bin/python3 -m pip install opencv-python requests torch flask


RUN mkdir -p /home/Documents
# That is for some kind of bug found on cv2

# Delete sources
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update -y && apt-get install libgl1 -y

COPY rest-cnn.py /home/Documents
COPY ResNet50_ImageNet_70_90_7_76GF_TensorRT_INT8_BATCH_32 /home/Documents/ResNet50_ImageNet_70_90_7_76GF_TensorRT_INT8_BATCH_32
#COPY client.py /home/Documents
COPY imagenet_class_index.json /home/Documents/imagenet_class_index.json
WORKDIR /home/Documents

ENV FLASK_APP=rest-cnn.py
EXPOSE 3001

# CMD source /workspace/setup.sh $DEVICE && /usr/bin/python3 rest-cnn.py
# CMD /usr/bin/python3 rest-cnn.py
CMD sleep infinity
