FROM tensorflow/tensorflow:2.8.0-gpu

RUN /usr/bin/python3 -m pip install opencv-python requests torch flask


RUN mkdir -p /home/Documents
# That is for some kind of bug found on cv2

# Delete sources
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update -y && apt-get install libgl1 -y

COPY best_UNET_v3_TensorRT_FP16_BATCH_1 /home/Documents/best_UNET_v3_TensorRT_FP16_BATCH_1
#COPY client.py /home/Documents
COPY rest-cnn.py /home/Documents
WORKDIR /home/Documents

ENV FLASK_APP=rest-cnn.py
EXPOSE 3001

# CMD source /workspace/setup.sh $DEVICE && /usr/bin/python3 rest-cnn.py
# CMD /usr/bin/python3 rest-cnn.py
CMD sleep infinity
