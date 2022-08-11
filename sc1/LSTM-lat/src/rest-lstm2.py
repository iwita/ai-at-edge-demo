#!/usr/bin/python3

#---------------------------------#
#- Author: Stamoulias Ioannis    -#
#- email: jstamoulias@gmail.com  -#
#---------------------------------#

import os
import sys
import numpy
import ctypes

# from sympy import true
from accel_libs.py_lib.accel_py import init_device, init_buffers, weights_biases, out_ptr, in_ptr, lstm, results, accel_end, read_file, write_file

from flask import Flask, request, Response
import json

initialized=False

def init():
    #The arguments exist for faster tests, 
    #everything can be hardcoded or passed 
    #through the final system
    global initialized
    try:
        #------------------------------------------------#
        #Only once at the beginning
        #------------------------------------------------#
        print("Initialize device")
        init_device(xclbin_name)
        print("Allocate and Map memories")
        init_buffers(timesteps)
        print("Convert weights and biases to short")
        wb_ptrs = weights_biases()
        print("Create output buffer")
        res_ptr = out_ptr()
        #------------------------------------------------#
        #At each iteration for processing new data
        #------------------------------------------------#
        valuein=read_file("./data/input.dat",int(timesteps)*int(runs_no))  #Input data from a file, or use your own variable


app=Flask(__name__)



@app.route('/api/infer',methods=['POST'])
def test():

    return ""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
