#!/usr/bin/python3

#---------------------------------#
#- Author: Stamoulias Ioannis    -#
#- email: jstamoulias@gmail.com  -#
#---------------------------------#

import os
import sys
import numpy
import ctypes
import linecache

accel = ctypes.CDLL('./accel_libs/accellib.so')

def parameters():
    MTIMESTEPS = 64
    FEATURES = 1
    IUNITS = 128
    OUNITS = 128
    iFRAQ = 10
    oFRAQ = 12
    iFRAQ_CONV = 1024
    oFRAQ_CONV = 4096
    return [MTIMESTEPS,FEATURES,IUNITS,OUNITS,iFRAQ,oFRAQ,iFRAQ_CONV,oFRAQ_CONV]

def init_device(xclbin_name):
    krnl_nm = xclbin_name #"krnl_lstm.xclbin"
    xclbin = ctypes.c_char_p(krnl_nm.encode('utf-8'))
    accel.device_init(xclbin)

def init_buffers(timesteps):
    PARAM=parameters()
    accel.lstm_io(int(timesteps), PARAM[1], PARAM[2], PARAM[3])

def read_file(fl_name, size):
    V = numpy.empty(size, float)
    with open(fl_name, "r") as fl:
        V = [float(next(fl)) for x in range(size)]
    return V

def read_file2(fl, size):
    V = numpy.empty(size, float)
    V = [float(next(fl)) for x in range(size)]
    return V

def write_file(fl_name,ar_var,ar_len):
    with open(fl_name, "w") as fl:
        for i in range(ar_len):
            fl.write(str(ar_var[i]))
            fl.write("\n")

def rmv_empty_ln(fl_name):
    with open(fl_name, "r") as flr, open(fl_name, "r+") as flw:
        for line in flr:
            if line.strip():
                flw.write(line)
        flw.truncate()

def fl2short_weights(W1, W2, rw, ln, fraq):
    accel.fixp.restype = ctypes.c_short
    #x = 0
    #y = 0
    buf_W = numpy.empty((2*rw*ln),ctypes.c_short)
    for i in range(2*rw*ln):
        #if y == (ln):
        #    if x == (rw-1):
        #        x = 0
        #    else:
        #        x = x + 1
        #    y = 0
        if (i<rw*ln): 
            buf_W[i] = accel.fixp(ctypes.c_float(W1[i]),fraq) #accel.fixp(ctypes.c_float(W1[x][y]),fraq)
        else: 
            buf_W[i] = accel.fixp(ctypes.c_float(W2[i-rw*ln]),fraq) #accel.fixp(ctypes.c_float(W2[x][y]),fraq)
       # y = y + 1
    return buf_W

def fl2short_biases(b1, b2, rw, fraq):
    accel.fixp.restype = ctypes.c_short
    x = 0
    buf_b = numpy.empty((2*rw),ctypes.c_short)
    for i in range(2*rw):
        if x == rw:
            x = 0
        if (i < rw): 
            buf_b[i] = accel.fixp(ctypes.c_float(b1[x]),fraq)
        else: 
            buf_b[i] = accel.fixp(ctypes.c_float(b2[x]),fraq)
        x = x + 1
    return buf_b

def fl2short_dense(Wd, bd, rw, fraq):
    accel.fixp.restype = ctypes.c_short
    buf_W = numpy.empty((rw),ctypes.c_short)
    buf_b = numpy.empty((1),ctypes.c_short)
    for i in range(rw):
        buf_W[i] = accel.fixp(ctypes.c_float(Wd[i]),fraq) #accel.fixp(ctypes.c_float(Wd[0][i]),fraq)
    buf_b[0] = accel.fixp(ctypes.c_float(bd[0]),fraq)
    return [buf_W, buf_b]

def weights_biases_accel(W1i,W2i,b1i,b2i,W1f,W2f,b1f,b2f,W1c,W2c,b1c,b2c,W1o,W2o,b1o,b2o,Wd,bd,rw,ln,fraq):
    shp = ctypes.POINTER(ctypes.c_short)
    buf_Wi = fl2short_weights(W1i, W2i, rw, ln, fraq)
    Wi_ptr = buf_Wi.ctypes.data_as(shp)
    buf_Wf = fl2short_weights(W1f, W2f, rw, ln, fraq)
    Wf_ptr = buf_Wf.ctypes.data_as(shp)
    buf_Wc = fl2short_weights(W1c, W2c, rw, ln, fraq)
    Wc_ptr = buf_Wc.ctypes.data_as(shp)
    buf_Wo = fl2short_weights(W1o, W2o, rw, ln, fraq)
    Wo_ptr = buf_Wo.ctypes.data_as(shp)
    buf_bi = fl2short_biases(b1i, b2i, rw, fraq)
    bi_ptr = buf_bi.ctypes.data_as(shp)
    buf_bf = fl2short_biases(b1f, b2f, rw, fraq)
    bf_ptr = buf_bf.ctypes.data_as(shp)
    buf_bc = fl2short_biases(b1c, b2c, rw, fraq)
    bc_ptr = buf_bc.ctypes.data_as(shp)
    buf_bo = fl2short_biases(b1o, b2o, rw, fraq)
    bo_ptr = buf_bo.ctypes.data_as(shp)
    [buf_W, buf_b] = fl2short_dense(Wd, bd, rw, fraq)
    W_ptr = buf_W.ctypes.data_as(shp)
    b_ptr = buf_b.ctypes.data_as(shp)
    return [Wi_ptr,bi_ptr,Wf_ptr,bf_ptr,Wc_ptr,bc_ptr,Wo_ptr,bo_ptr,W_ptr,b_ptr]

def weights_biases():
    PARAM=parameters()
    Wi1_f=read_file("./weights_biases/Wi1",PARAM[2]*(PARAM[2]+PARAM[3]))
    Wf1_f=read_file("./weights_biases/Wf1",PARAM[2]*(PARAM[2]+PARAM[3]))
    Wc1_f=read_file("./weights_biases/Wc1",PARAM[2]*(PARAM[2]+PARAM[3]))
    Wo1_f=read_file("./weights_biases/Wo1",PARAM[2]*(PARAM[2]+PARAM[3]))
    Wi2_f=read_file("./weights_biases/Wi2",PARAM[2]*(PARAM[2]+PARAM[3]))
    Wf2_f=read_file("./weights_biases/Wf2",PARAM[2]*(PARAM[2]+PARAM[3]))
    Wc2_f=read_file("./weights_biases/Wc2",PARAM[2]*(PARAM[2]+PARAM[3]))
    Wo2_f=read_file("./weights_biases/Wo2",PARAM[2]*(PARAM[2]+PARAM[3]))
    bi1_f=read_file("./weights_biases/bi1",PARAM[2])
    bf1_f=read_file("./weights_biases/bf1",PARAM[2])
    bc1_f=read_file("./weights_biases/bc1",PARAM[2])
    bo1_f=read_file("./weights_biases/bo1",PARAM[2])
    bi2_f=read_file("./weights_biases/bi2",PARAM[2])
    bf2_f=read_file("./weights_biases/bf2",PARAM[2])
    bc2_f=read_file("./weights_biases/bc2",PARAM[2])
    bo2_f=read_file("./weights_biases/bo2",PARAM[2])
    W_f=read_file("./weights_biases/W",PARAM[2])
    b_f=read_file("./weights_biases/b",1)
    ptrs = weights_biases_accel(Wi1_f,Wi2_f,bi1_f,bi2_f,Wf1_f,Wf2_f,bf1_f,bf2_f,Wc1_f,Wc2_f,bc1_f,bc2_f,Wo1_f,Wo2_f,bo1_f,bo2_f,W_f,b_f,PARAM[2],(PARAM[2]+PARAM[3]),PARAM[4])
    return ptrs

def fl2short_input(data, max_steps, timesteps, features, fraq):
    accel.fixp.restype = ctypes.c_short
    buf_data = numpy.empty((max_steps*features),ctypes.c_short)
    for i in range(max_steps*features):	
        if (i < timesteps*features): 
            buf_data[i] = accel.fixp(ctypes.c_float(data[i]),fraq)
        else:  
            buf_data[i] = 0
    return buf_data

def out_ptr():
    PARAM = parameters()
    shp = ctypes.POINTER(ctypes.c_short)
    buf_res = numpy.zeros((PARAM[0]*PARAM[1]),ctypes.c_short)
    res_ptr = buf_res.ctypes.data_as(shp)
    return res_ptr

def in_ptr(data_in, timesteps):
    PARAM = parameters()
    shp = ctypes.POINTER(ctypes.c_short)
    buf_data = fl2short_input(data_in, PARAM[0], int(timesteps), PARAM[1], PARAM[4])
    data_ptr = buf_data.ctypes.data_as(shp)
    return data_ptr

def lstm(data_ptr, res_ptr, wb_ptrs, timesteps, flag):
    PARAM = parameters()
    accel.lstm_accel(data_ptr,res_ptr,wb_ptrs[0],wb_ptrs[1],wb_ptrs[2],wb_ptrs[3],wb_ptrs[4],wb_ptrs[5],wb_ptrs[6],wb_ptrs[7],wb_ptrs[8],wb_ptrs[9],int(timesteps),flag,PARAM[1],PARAM[2],PARAM[3])

def results(res_ptr, timesteps):
    PARAM = parameters()
    R = numpy.empty(int(timesteps), float)
    for i in range(int(timesteps)):
            R[i] = (res_ptr[i]/PARAM[7])
    return R

def print_res(res_ptr, res, timesteps):
    PARAM = parameters()
    #res = results(res_ptr, timesteps)
    if(res[(int(timesteps)-1)]>0.5):
        ret ="Attack detected : " + str(res[(int(timesteps)-1)])
        # print("Attack detected : ",res[(int(timesteps)-1)])
    else:
        ret ="Normal process : " + str(res[(int(timesteps)-1)])
    print(ret)
    return ret


def accel_end():
    accel.lstm_end()

