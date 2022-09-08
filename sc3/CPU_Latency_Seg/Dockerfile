FROM intel/intel-optimized-tensorflow

RUN pip install opencv-python requests flask
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN mkdir -p /home/Documents
# That is for some kind of bug found on cv2

COPY best_UNET_v3.tflite /home/Documents
#COPY client.py /home/Documents
COPY rest-cnn.py /home/Documents
WORKDIR /home/Documents

ENV FLASK_APP=rest-cnn.py
EXPOSE 3000

ENV CORES=8
# CMD source /workspace/setup.sh $DEVICE && /usr/bin/python3 rest-cnn.py
# CMD /usr/bin/python3 rest-cnn.py
CMD sleep infinity
