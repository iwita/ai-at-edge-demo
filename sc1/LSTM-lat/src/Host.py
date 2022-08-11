#!/usr/bin/python3

#---------------------------------#
#- Author: Stamoulias Ioannis    -#
#- email: jstamoulias@gmail.com  -#
#---------------------------------#

import os
import sys
import numpy
import ctypes
from accel_libs.py_lib.accel_py import init_device, init_buffers, weights_biases, out_ptr, in_ptr, lstm, results, accel_end, read_file, write_file

def main(args):
    #The arguments exist for faster tests, 
    #everything can be hardcoded or passed 
    #through the final system
    if(len(args)==1):
        [name] = args
        print("No XCLBIN SET")
        print("FAILED TEST")
        return -1
    if(len(args)==2):
    	[name, xclbin_name] = args
    	timesteps = 50
    	runs_no = 1
    if(len(args)==3):
    	[name, xclbin_name, timesteps] = args
    	runs_no = 1
    if(len(args)==4):
    	[name, xclbin_name, timesteps, runs_no] = args

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
        for i in range(int(runs_no)):
            print("---- Run: ", i," ----")
            print("Convert input to short")
            data_ptr = in_ptr(valuein[(i*int(timesteps)):(((i+1)*int(timesteps)))], timesteps)
            print("Run accelerator")
            lstm(data_ptr, res_ptr, wb_ptrs, timesteps, 1)
            print("Convert Results")
            res = results(res_ptr, timesteps)
            print(res)
            write_file("./data/output.dat",res,len(res))
        #------------------------------------------------#
        #Only once at the end
        #------------------------------------------------#
        print("Release buffers")
        accel_end()
        return 0

    except OSError as o:
        print(o)
        print("FAILED TEST")
        return -o.errno

    except AssertionError as a:
        print(a)
        print("FAILED TEST")
        return -1
    except Exception as e:
        print(e)
        print("FAILED TEST")
        return -1

if __name__ == "__main__":
    result = main(sys.argv)
    sys.exit(result)

