FROM ubuntu:18.04

#Install dependencies
RUN apt-get update; apt-get install -y wget \
    python3-pip

#Dowload XRT installation packages from Xilinx page
RUN wget -cO - https://www.xilinx.com/bin/public/openDownload?filename=xrt_202110.2.11.634_18.04-amd64-xrt.deb > /root/xrt_202110.2.11.634_18.04-amd64-xrt.deb

#Install XRT
RUN apt-get update; apt-get install -y /root/xrt_202110.2.11.634_18.04-amd64-xrt.deb

RUN apt-get install vim -y
#Install python packages
#RUN apt install python3-pip
RUN pip3 install numpy Flask

#Copy executable and xclbin files
COPY ./accel_libs ./lstm_app/accel_libs
# COPY ./data ./lstm_app/data
COPY ./weights_biases ./lstm_app/weights_biases
COPY ./Host.py ./lstm_app/Host.py
COPY ./rest-lstm.py ./lstm_app/rest-lstm.py
COPY ./krnl_lstm.xclbin ./lstm_app/krnl_lstm.xclbin

WORKDIR /lstm_app

ENV FLASK_APP=rest-lstm.py
#Set the enviroment variables for xrt use
ENV XILINX_XRT=/opt/xilinx/xrt
ENV PATH=/opt/xilinx/xrt/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:./lstm_app/accel_libs/accellib.so
ENV PYTHONPATH=/opt/xilinx/xrt/python

RUN chmod u+x ./rest-lstm.py

EXPOSE 5000
# CMD while true; do sleep 100000s; done
CMD ./rest-lstm.py