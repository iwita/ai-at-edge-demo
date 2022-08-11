#*******************************************************************************
#Vendor: Xilinx 
#Associated Filename: common.mk
#Purpose: Common Makefile for VITIS Compilation
#
#*******************************************************************************
#Copyright (C) 2015-2019 XILINX, Inc.
#
#This file contains confidential and proprietary information of Xilinx, Inc. and 
#is protected under U.S. and international copyright and other intellectual 
#property laws.
#
#DISCLAIMER
#This disclaimer is not a license and does not grant any rights to the materials 
#distributed herewith. Except as otherwise provided in a valid license issued to 
#you by Xilinx, and to the maximum extent permitted by applicable law: 
#(1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX 
#HEREBY DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, 
#INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, OR 
#FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether 
#in contract or tort, including negligence, or under any other theory of 
#liability) for any loss or damage of any kind or nature related to, arising under 
#or in connection with these materials, including for any direct, or any indirect, 
#special, incidental, or consequential loss or damage (including loss of data, 
#profits, goodwill, or any type of loss or damage suffered as a result of any 
#action brought by a third party) even if such damage or loss was reasonably 
#foreseeable or Xilinx had been advised of the possibility of the same.
#
#CRITICAL APPLICATIONS
#Xilinx products are not designed or intended to be fail-safe, or for use in any 
#application requiring fail-safe performance, such as life-support or safety 
#devices or systems, Class III medical devices, nuclear facilities, applications 
#related to the deployment of airbags, or any other applications that could lead 
#to death, personal injury, or severe property or environmental damage 
#(individually and collectively, "Critical Applications"). Customer assumes the 
#sole risk and liability of any use of Xilinx products in Critical Applications, 
#subject only to applicable laws and regulations governing limitations on product 
#liability. 
#
#THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT 
#ALL TIMES.
#
#*******************************************************************************
SHELL = /bin/bash
VPATH = ./

#supported flow: cpu_emu, hw_emu, hw
CC = g++
CLCC = v++

ifeq ($(XDEVICE_REPO_PATH),)
#no device repo path set. do nothing
    DEVICE_REPO_OPT = 
else
    DEVICE_REPO_OPT = --xp prop:solution.platform_repo_paths=${XDEVICE_REPO_PATH} 
endif

HOST_CFLAGS += -I${XILINX_XRT}/include/ -I${XILINX_VIVADO}/include/ -lOpenCL -lstdc++
HOST_LFLAGS += -L${XILINX_XRT}/lib/ -lrt -pthread -lxilinxopencl
CLCC_OPT += $(CLCC_OPT_LEVEL) ${DEVICE_REPO_OPT} --platform ${XDEVICE} ${KERNEL_DEFS} ${KERNEL_INCS}

ifeq (${KEEP_TEMP},1)
    CLCC_OPT += -s
endif

ifeq (${KERNEL_DEBUG},1)
    CLCC_OPT += -g
endif

ifeq (${SDA_FLOW},cpu_emu)
	EMU_MODE=sw_emu
else ifeq (${SDA_FLOW},hw_emu)
	EMU_MODE=hw_emu
endif

CLCC_OPT += --kernel ${KERNEL_NAME}
OBJECTS := $(HOST_SRCS:.cpp=.o)

.PHONY: all

all: run

host: ${HOST_EXE_DIR}/${HOST_EXE}

xbin_cpu_em:
	make SDA_FLOW=cpu_emu xbin -f vitis.mk

xbin_hw_em:
	make SDA_FLOW=hw_emu xbin -f vitis.mk

xbin_hw :
	make SDA_FLOW=hw xbin -f vitis.mk

xbin: ${XCLBIN}

run_cpu_em: 
	make SDA_FLOW=cpu_emu run_em -f vitis.mk

run_hw_em: 
	make SDA_FLOW=hw_emu run_em -f vitis.mk

run_hw : 
	make SDA_FLOW=hw run_hw_int -f vitis.mk

run_em: xconfig host xbin
	XCL_EMULATION_MODE=${EMU_MODE} ${HOST_EXE_DIR}/${HOST_EXE} ${HOST_ARGS}

run_hw_int : host xbin_hw
	${HOST_EXE_DIR}/${HOST_EXE} ${HOST_ARGS}

estimate : 
	${CLCC} -c -t hw_emu --platform ${XDEVICE} --report estimate ${KERNEL_SRCS}

xconfig : emconfig.json

emconfig.json :
	emconfigutil --platform ${XDEVICE} ${DEVICE_REPO_OPT} --od .

${HOST_EXE_DIR}/${HOST_EXE} : ${OBJECTS}
	${CC} ${HOST_LFLAGS} ${OBJECTS} -o $@

${XCLBIN}:
	${CLCC} -c ${CLCC_OPT} ${KERNEL_SRCS} -o $(XO)
	${CLCC} -l ${CLCC_OPT} $(XO) -o $(XCLBIN)

%.o: %.cpp
	${CC} ${HOST_CFLAGS} -c $< -o $@

clean:
	${RM} -rf ${HOST_EXE} ${OBJECTS} emconfig.json .Xil *v++* *.log *.link_summary *.compile_summary _x/ *.pb *.xo

cleanall: clean
	${RM} -rf *.xclbin profile_summary.* TempConfig *.jou _vimage/ bin_vadd_cpu_emu.x* bin_vadd_hw_emu.x* bin_vadd_hw.x*


help:
	@echo "Compile and run CPU emulation"
	@echo "make -f vitis.mk run_cpu_em"
	@echo ""
	@echo "Compile and run hardware emulation"
	@echo "make -f vitis.mk run_hw_em"
	@echo ""
	@echo "Compile host executable only"
	@echo "make -f vitis.mk host"
	@echo ""
	@echo "Compile XCLBIN file for system run only"
	@echo "make -f vitis.mk xbin_hw"
	@echo ""
	@echo "Compile and run CPU emulation using xilinx_u200_xdma_201920_1 XSA"
	@echo "make -f vitis.mk XDEVICE=xilinx_u200_xdma_201920_1 run_cpu_em"
	@echo ""
	@echo "Clean working diretory"
	@echo "make -f vitis.mk clean"
	@echo ""
	@echo "Super clean working directory"
	@echo "make -f vitis.mk cleanall"
