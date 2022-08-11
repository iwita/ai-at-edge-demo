#!/usr/bin/python3

#---------------------------------#
#- Author: Stamoulias Ioannis    -#
#- email: jstamoulias@gmail.com  -#
#---------------------------------#

from fileinput import filename
import os
import sys
import numpy
import ctypes
from io import BufferedReader
from werkzeug.utils import secure_filename


# from sympy import true
from accel_libs.py_lib.accel_py import init_device, init_buffers, weights_biases, out_ptr, in_ptr, lstm, results, accel_end, read_file, read_file2, write_file, print_res

from flask import Flask, request, Response, jsonify
import logging

import json
import datetime

logging.basicConfig(level=logging.DEBUG)

initialized=False
timesteps=50
xclbin_name='krnl_lstm.xclbin'
runs_no=1
ALLOWED_EXTENSIONS = {'dat'}
avg_elapsed_time_d = 0.0
avg_elapsed_time_e = 0.0
avg_total_time=0.0
elapsed_time_i= 0

#res_ptr
#wb_ptrs

def init(xclbin_name,timesteps):
    global initialized
    try:
        #------------------------------------------------#
        #Only once at the beginning
        #------------------------------------------------#
        st=datetime.datetime.now()
        print("Initialize device")
        init_device(xclbin_name)
        print("Allocate and Map memories")
        init_buffers(timesteps)
        print("Convert weights and biases to short")
        global wb_ptrs
        wb_ptrs = weights_biases()
        print("Create output buffer")
        global res_ptr
        res_ptr = out_ptr()
        initialized=True
        et = datetime.datetime.now()
        global elapsed_time_i
        elapsed_time_i = et - st
        # print('Initialize time:', elapsed_time_i.total_seconds()*1000, 'ms')
        app.logger.info('Initialize time:' + str(elapsed_time_i.total_seconds()*1000) + 'ms')
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

def end():
    print("Release buffers")
    accel_end()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app=Flask(__name__)


@app.route('/api/infer',methods=['POST'])
def test():
    # app.logger.info('Hello')  
    # print("Hello2",file=sys.stdout)  
    global elapsed_time_i
    global avg_elapsed_time_d
    global avg_elapsed_time_e
    global avg_total_time
    avg_elapsed_time_d = 0.0
    avg_elapsed_time_e = 0.0
    avg_total_time=0.0


	# check if the post request has the file part
    # filename='input.dat'
    if 'file' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    file=request.files['file']
    if file.filename == '':
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    if not allowed_file(file.filename):
        resp=jsonify({'message':"Allowed File Types .dat"})    
        resp.status_code = 400
        return resp
    elif file:    
        try:
            print(file.filename)

            # Convert from FileStorage (Flask related object) to FileObject to read
            file = BufferedReader(file)
            # valuein=read_file(file.filename,int(timesteps)*int(runs_no))  #Input data from a file, or use your own variable
            valuein=read_file2(file,int(timesteps)*int(runs_no)) 
            for i in range(int(runs_no)):
                #print("---- Run: ", i," ----")
                #print("Convert input to short")
                st = datetime.datetime.now()
                data_ptr = in_ptr(valuein[(i*int(timesteps)):(((i+1)*int(timesteps)))], timesteps)
                et = datetime.datetime.now()
                elapsed_time_d = et - st
                #print("Run accelerator")
                st = datetime.datetime.now()
                if(i==0):
                    lstm(data_ptr, res_ptr, wb_ptrs, timesteps, 1)
                else:
                    lstm(data_ptr, res_ptr, wb_ptrs, timesteps, 0)
                et = datetime.datetime.now()
                elapsed_time_e = et - st
                #print("Convert Results")
                st = datetime.datetime.now()
                res = results(res_ptr, timesteps)
                et = datetime.datetime.now()
                elapsed_time_d2 = et - st
                #print("Print Results")
                ret = print_res(res_ptr, res, timesteps)
                write_file("./data/output.dat",res,len(res))
                avg_elapsed_time_d=avg_elapsed_time_d+elapsed_time_d.total_seconds()*1000+elapsed_time_d2.total_seconds()*1000
                avg_elapsed_time_e=avg_elapsed_time_e+elapsed_time_e.total_seconds()*1000
                avg_total_time=avg_total_time+elapsed_time_e.total_seconds()*1000+elapsed_time_d.total_seconds()*1000+elapsed_time_d2.total_seconds()*1000

    
            # print('Data preparation time:', avg_elapsed_time_d/int(runs_no), 'ms')
            app.logger.info(ret)
            app.logger.info('Data preparation time: %.2fms', avg_elapsed_time_d/int(runs_no))
            # app.logger.info('Data preparation time:', avg_elapsed_time_d/int(runs_no), 'ms')
            # print('Execution time:', avg_elapsed_time_e/int(runs_no), 'ms')
            app.logger.info('Execution time: %.2fms', avg_elapsed_time_e/int(runs_no))
            app.logger.info('Total E2E Latency: %.2fms', avg_total_time/int(runs_no))
            app.logger.info('Total throughput: %.2frps', 1000/avg_total_time)
            # app.logger.info('Execution time:', avg_elapsed_time_e/int(runs_no), 'ms')

        except OSError as o:
            print(o)
            print("FAILED TEST")
            resp = jsonify({'message' : 'Error:'+str(o.errno)})
            resp.status_code = 400
            return resp
        except AssertionError as a:
            print(a)
            print("FAILED TEST")
            resp = jsonify({'message' : 'Error:'+str(-1)})
            resp.status_code = 400
            return resp
            # return -1
        except Exception as e:
            print(e)
            print("FAILED TEST")
            resp = jsonify({'message' : 'Error:'+str(-1)})
            resp.status_code = 400
            return resp
            # return -1

    return Response(response=json.dumps({'res':ret}),status=200,mimetype="application/json")
    # return Response(response=json.dumps({"res":"ok"}),status=200,mimetype="application/json")

@app.route('/api/test2',methods=['POST'])
def test2():

    return Response(response=json.dumps({"res":"ok"}),status=200,mimetype="application/json")


if __name__ == "__main__":
    # global timesteps
    # global runs_no
    if(len(sys.argv)==1):
        name = sys.argv[0]
        print("No XCLBIN SET")
        print("FAILED TEST")
        # return -1
    if(len(sys.argv)==2):
        xclbin_name = sys.argv[1]
        # timesteps=50
    	# runs_no=1
    if(len(sys.argv)==3):
        [name, xclbin_name, timesteps] = sys.argv
    	# runs_no = 1
    if(len(sys.argv)==4):
        [name, xclbin_name, timesteps, runs_no] = sys.argv
        
    init(xclbin_name,timesteps)
    app.run(host="0.0.0.0", port=5000)
