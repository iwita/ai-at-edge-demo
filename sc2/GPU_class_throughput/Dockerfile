FROM tensorflow/tensorflow:2.8.0-gpu
# FROM iwita/vitis-ai-base

#ENV DEVICE=U280_H
# Heavy part. Keep it on the top to avoid re-installing
# RUN /usr/bin/python3 -m pip install --upgrade pip
# RUN /usr/bin/python3 -m pip install tensorflow
# RUN /usr/bin/python3 -m pip install --upgrade tensorflow


# USER vitis-ai-user
# TOBEREMOVED
# For the client

RUN /usr/bin/python3 -m pip install opencv-python requests torch flask


RUN mkdir -p /home/Documents
# That is for some kind of bug found on cv2

# Delete sources
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update -y && apt-get install libgl1 -y

COPY rest-cnn.py /home/Documents
# COPY test_images /workspace/test_images
# COPY customcnn.xmodel /workspace
COPY ResNet50_ImageNet_70_90_7_76GF_TensorRT_INT8_BATCH_128 /home/Documents/ResNet50_ImageNet_70_90_7_76GF_TensorRT_INT8_BATCH_128
# COPY setup.sh /workspace
COPY client.py /home/Documents

WORKDIR /home/Documents

# RUN chown -R vitis-ai-user:vitis-ai-group /workspace

# RUN pip3 install -r requirements.txt 

ENV FLASK_APP=rest-cnn.py
EXPOSE 3000

# CMD source /workspace/setup.sh $DEVICE && /usr/bin/python3 rest-cnn.py
# CMD /usr/bin/python3 rest-cnn.py
CMD sleep infinity
