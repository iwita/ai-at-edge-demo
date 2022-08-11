#!/usr/bin/python3

#---------------------------------#
#- Author: Stamoulias Ioannis    -#
#- email: jstamoulias@gmail.com  -#
#---------------------------------#

import os
import sys
import numpy
import ctypes
from accel_libs.py_lib.accel_py import init_device, init_buffers, weights_biases, out_ptr, in_ptr, lstm, results, print_res, accel_end, read_file, write_file

import datetime

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
    	batch = 256
    	timesteps = 50
    	runs_no = 1
    if(len(args)==3):
    	[name, xclbin_name, batch] = args
    	timesteps = 50
    	runs_no = 1
    if(len(args)==4):
    	[name, xclbin_name, batch, timesteps] = args
    	runs_no = 1
    if(len(args)==5):
    	[name, xclbin_name, batch, timesteps, runs_no] = args

    try:
        #------------------------------------------------#
        #Only once at the beginning
        #------------------------------------------------#
        st = datetime.datetime.now()
        #print("Initialize device")
        init_device(xclbin_name)
        #print("Allocate and Map memories")
        init_buffers(batch, timesteps)
        #print("Convert weights and biases to short")
        wb_ptrs = weights_biases()
        #print("Create output buffer")
        res_ptr = out_ptr(batch)
        et = datetime.datetime.now()
        elapsed_time_i = et - st
        #print('Initialize time:', elapsed_time_i.total_seconds()*1000, 'ms')
        #------------------------------------------------#
        #At each iteration for processing new data
        #------------------------------------------------#
        avg_elapsed_time_d=0.0
        avg_elapsed_time_e=0.0
        valuein=read_file("./data/input.dat",int(batch)*int(timesteps)*int(runs_no))  #Input data from a file, or use your own variable
        for i in range(int(runs_no)):
            #print("---- Run: ", i," ----")
            #print("Convert input to short")
            st = datetime.datetime.now()
            data_ptr = in_ptr(valuein[(i*int(batch)*int(timesteps)):(((i+1)*int(batch)*int(timesteps)))], batch, timesteps)
            et = datetime.datetime.now()
            elapsed_time_d = et - st
            #print("Run accelerator")
            st = datetime.datetime.now()
            if(i==0):
                lstm(data_ptr, res_ptr, wb_ptrs, batch, timesteps, 1)
            else:
                lstm(data_ptr, res_ptr, wb_ptrs, batch, timesteps, 0)
            et = datetime.datetime.now()
            elapsed_time_e = et - st
            #print("Convert Results")
            st = datetime.datetime.now()
            res = results(res_ptr, batch, timesteps)
            et = datetime.datetime.now()
            elapsed_time_d2 = et - st
            #print("Print Results")
            print_res(res_ptr, res, batch, timesteps)
            write_file("./data/output.dat",res,len(res))

            avg_elapsed_time_d=avg_elapsed_time_d+(elapsed_time_d.total_seconds()*1000+elapsed_time_d2.total_seconds()*1000)
            avg_elapsed_time_e=avg_elapsed_time_e+elapsed_time_e.total_seconds()*1000
            #print('Data preparation time:', elapsed_time_d.total_seconds()*1000+elapsed_time_d2.total_seconds()*1000, 'ms')
            #print('Execution time:', elapsed_time_e.total_seconds()*1000, 'ms')
        #------------------------------------------------#
        #Only once at the end
        #------------------------------------------------#
        #print("Release buffers")
        accel_end()
        print('Initialize time:', elapsed_time_i.total_seconds()*1000, 'ms')
        print('Data preparation time:', avg_elapsed_time_d/int(runs_no), 'ms')
        print('Execution time:', avg_elapsed_time_e/int(runs_no), 'ms')
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

