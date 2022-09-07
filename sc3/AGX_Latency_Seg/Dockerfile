FROM nvcr.io/nvidia/l4t-tensorflow:r32.7.1-tf2.7-py3


# Use this on host if docker buildx build errors like this : Error while loading /usr/sbin/dpkg-split: No such file or directory 
# docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

RUN apt-get update -y
RUN pip3 install --upgrade pip
# We no longer need torch. The following would install it though
#RUN apt-get install libopenblas-base -y
#RUN apt-get install libopenmpi-dev -y
#RUN wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
#RUN pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl

RUN pip3 install onnx
RUN wget https://nvidia.box.com/shared/static/pmsqsiaw4pg9qrbeckcbymho6c01jj4z.whl -O onnxruntime_gpu-1.11.0-cp36-cp36m-linux_aarch64.whl
RUN pip3 install onnxruntime_gpu-1.11.0-cp36-cp36m-linux_aarch64.whl
RUN rm onnxruntime_gpu-1.11.0-cp36-cp36m-linux_aarch64.whl

# Maybe dont do tf2onnx on the AGX because it certainly needs much memory
# RUN pip3 install tf2onnx

RUN pip3 install opencv-python
RUN pip3 install flask
RUN pip3 install requests
#
# RUN pip3 install -U flatbuffers
# RUN export LD_PRELOAD=/usr/local/lib/python3.6/dist-packages/torch/lib/libgomp-d22c30c5.so.1
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

RUN mkdir -p /home/Documents


COPY best_UNET_v3_B1.onnx /home/Documents/best_UNET_v3_B1.onnx
COPY calibration.flatbuffers /home/Documents/calibration.flatbuffers
#COPY client.py /home/Documents/client.py
COPY rest-cnn.py /home/Documents
WORKDIR /home/Documents
 
ENV FLASK_APP=rest-cnn.py
EXPOSE 3000

# CMD source /workspace/setup.sh $DEVICE && /usr/bin/python3 rest-cnn.py
# CMD /usr/bin/python3 rest-cnn.py
CMD sleep infinity
