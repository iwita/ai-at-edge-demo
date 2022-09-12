FROM xilinx/vitis-ai:1.4.1.978
# FROM iwita/vitis-ai-base

ENV DEVICE=U280_L
# Heavy part. Keep it on the top to avoid re-installing
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN /usr/bin/python3 -m pip install tensorflow
RUN /usr/bin/python3 -m pip install --upgrade tensorflow


# USER vitis-ai-user
# TOBEREMOVED
# For the client

RUN /usr/bin/python3 -m pip install opencv-python flask requests

WORKDIR /workspace

# COPY test_images /workspace/test_images
COPY customcnn.xmodel /workspace
COPY setup.sh /workspace
COPY rest-cnn.py /workspace

WORKDIR /workspace

RUN chown -R vitis-ai-user:vitis-ai-group /workspace

# RUN pip3 install -r requirements.txt 

ENV FLASK_APP=rest-cnn.py
EXPOSE 3001

#CMD source /workspace/setup.sh $DEVICE && /usr/bin/python3 rest-cnn.py
CMD source /workspace/setup.sh $DEVICE
CMD sleep infinity
