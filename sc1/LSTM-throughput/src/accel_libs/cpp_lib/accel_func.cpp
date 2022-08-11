//---------------------------------//
//- Author: Stamoulias Ioannis    -//
//- email: jstamoulias@gmail.com  -//
//---------------------------------//

#include <iostream>
#include <fstream>
#include <math.h>

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#include <CL/cl2.hpp>

typedef struct{ 
	//Device variables
	cl::Context context;
	cl::CommandQueue command_queue;
	cl::Kernel krnl;
	//Kernel variables
	cl::Buffer buffer_data;
	cl::Buffer buffer_Wi;
	cl::Buffer buffer_bi;
	cl::Buffer buffer_Wf;
	cl::Buffer buffer_bf;
	cl::Buffer buffer_Wc;
	cl::Buffer buffer_bc;
	cl::Buffer buffer_Wo;
	cl::Buffer buffer_bo;
	cl::Buffer buffer_W;
	cl::Buffer buffer_b;
	cl::Buffer buffer_res;
}krnl_world;
krnl_world world;

extern "C"{

void device_init(char* krnl_name)
{
	// Creates a vector of DATA_SIZE elements with an initial value of 10 and 32
	// using customized allocator for getting buffer alignment to 4k boundary
	std::vector<cl::Device> devices;
	cl::Device device;
	std::vector<cl::Platform> platforms;
	bool found_device = false;

	//traversing all Platforms To find Xilinx Platform and targeted
	//Device in Xilinx Platform
	cl::Platform::get(&platforms);
	for(size_t i = 0; (i < platforms.size() ) & (found_device == false) ;i++){
		cl::Platform platform = platforms[i];
		std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
		if ( platformName == "Xilinx"){
			devices.clear();
			platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
		if (devices.size()){
			device = devices[0];
			found_device = true;
			break;
		}
		}
	}
	if (found_device == false){
	   std::cout << "Error: Unable to find Target Device "
		   << device.getInfo<CL_DEVICE_NAME>() << std::endl;
	   //return EXIT_FAILURE;
	   exit(EXIT_FAILURE);
	}

	// Creating Context and Command Queue for selected device
        world.context = cl::Context(device); //CREATES THE SEG FAULT IN PYTHON - DOES NOT CLOSES CORRECT
	world.command_queue = cl::CommandQueue(world.context, device, CL_QUEUE_PROFILING_ENABLE);

	char* xclbinFilename = krnl_name;
	// Load xclbin
	std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
	bin_file.seekg (0, bin_file.end);
	unsigned nb = bin_file.tellg();
	bin_file.seekg (0, bin_file.beg);
	char *buf = new char [nb];
	bin_file.read(buf, nb);
	std::string name_all = krnl_name;
	size_t pos = name_all.find_first_of('.');
	std::string name = name_all.substr(0,pos);
	int size_name = name.length();
	char name_c[size_name+1];
	strcpy(name_c, name.c_str());

	// Creating Program from Binary File
	cl::Program::Binaries bins;
	bins.push_back({buf,nb});
	devices.resize(1);
	cl::Program program(world.context, devices, bins);

	// This call will get the kernel object from program. A kernel is an
	// OpenCL function that is executed on the FPGA.
	world.krnl = cl::Kernel(program, name_c);
}

void lstm_io(int time_steps, int max_steps, int features, int iunits, int ounits, int mx_batch)
{
	size_t size_in_bytes = mx_batch*max_steps*features * sizeof(short int);
	size_t size_out_bytes = mx_batch*max_steps*features * sizeof(short int);
	size_t weights_in_bytes = 2*iunits*(iunits+ounits) * sizeof(short int);
	size_t bias_in_bytes = 2*iunits * sizeof(short int);
	size_t w_in_bytes = features*ounits * sizeof(short int);
	size_t b_in_bytes = features * sizeof(short int);
	cl_int err;

	// These commands will allocate memory on the Device.
	world.buffer_data = cl::Buffer(world.context, CL_MEM_READ_ONLY, size_in_bytes, nullptr, &err);
	world.buffer_Wi = cl::Buffer(world.context, CL_MEM_READ_ONLY, weights_in_bytes, nullptr, &err);
	world.buffer_bi = cl::Buffer(world.context, CL_MEM_READ_ONLY, bias_in_bytes, nullptr, &err);
	world.buffer_Wf = cl::Buffer(world.context, CL_MEM_READ_ONLY, weights_in_bytes, nullptr, &err);
	world.buffer_bf = cl::Buffer(world.context, CL_MEM_READ_ONLY, bias_in_bytes, nullptr, &err);
	world.buffer_Wc = cl::Buffer(world.context, CL_MEM_READ_ONLY, weights_in_bytes, nullptr, &err);
	world.buffer_bc = cl::Buffer(world.context, CL_MEM_READ_ONLY, bias_in_bytes, nullptr, &err);
	world.buffer_Wo = cl::Buffer(world.context, CL_MEM_READ_ONLY, weights_in_bytes, nullptr, &err);
	world.buffer_bo = cl::Buffer(world.context, CL_MEM_READ_ONLY, bias_in_bytes, nullptr, &err);
	world.buffer_W = cl::Buffer(world.context, CL_MEM_READ_ONLY, w_in_bytes, nullptr, &err);
	world.buffer_b = cl::Buffer(world.context, CL_MEM_READ_ONLY, b_in_bytes, nullptr, &err);	
	world.buffer_res = cl::Buffer(world.context, CL_MEM_WRITE_ONLY, size_out_bytes, nullptr, &err);

	int timesteps_hw=time_steps;
	//set the kernel Arguments
	int narg=0;
	world.krnl.setArg(narg++,world.buffer_data);
	world.krnl.setArg(narg++,world.buffer_Wi);
	world.krnl.setArg(narg++,world.buffer_bi);
	world.krnl.setArg(narg++,world.buffer_Wf);
	world.krnl.setArg(narg++,world.buffer_bf);
	world.krnl.setArg(narg++,world.buffer_Wc);
	world.krnl.setArg(narg++,world.buffer_bc);
	world.krnl.setArg(narg++,world.buffer_Wo);
	world.krnl.setArg(narg++,world.buffer_bo);
	world.krnl.setArg(narg++,world.buffer_W);
	world.krnl.setArg(narg++,world.buffer_b);
	world.krnl.setArg(narg++,world.buffer_res);
	world.krnl.setArg(narg++,timesteps_hw);
	world.krnl.setArg(narg++,mx_batch);
	// Data will be migrated to kernel space
	world.command_queue.enqueueMigrateMemObjects({world.buffer_data, world.buffer_Wi, world.buffer_bi, world.buffer_Wf, world.buffer_bf, world.buffer_Wc, world.buffer_bc, world.buffer_Wo, world.buffer_bo, world.buffer_W, world.buffer_b, world.buffer_res}, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, nullptr, nullptr);
	world.command_queue.finish();
}

void lstm_accel(short int *input, short int *output, short int *Wi, short int *bi, short int *Wf, short int *bf, short int *Wc, short int *bc, short int *Wo, short int *bo, short int *W, short int *b, int time_steps, int max_steps, int init, int features, int iunits, int ounits, int mx_batch)
{
	// Compute the size of array in bytes
	size_t size_in_bytes = mx_batch*max_steps*features * sizeof(short int);
	size_t size_out_bytes = mx_batch*max_steps*features * sizeof(short int);
	size_t weights_in_bytes = 2*iunits*(iunits+ounits) * sizeof(short int);
	size_t bias_in_bytes = 2*iunits * sizeof(short int);
	size_t w_in_bytes = features*ounits * sizeof(short int);
	size_t b_in_bytes = features * sizeof(short int);

	//Write input data to buffer
	world.command_queue.enqueueWriteBuffer(world.buffer_data, CL_TRUE, 0, size_in_bytes, input, nullptr, nullptr);
	if(init==1)
	{
		world.command_queue.enqueueWriteBuffer(world.buffer_Wi, CL_TRUE, 0, weights_in_bytes, Wi, nullptr, nullptr);
		world.command_queue.enqueueWriteBuffer(world.buffer_bi, CL_TRUE, 0, bias_in_bytes, bi, nullptr, nullptr);
		world.command_queue.enqueueWriteBuffer(world.buffer_Wf, CL_TRUE, 0, weights_in_bytes, Wf, nullptr, nullptr);
		world.command_queue.enqueueWriteBuffer(world.buffer_bf, CL_TRUE, 0, bias_in_bytes, bf, nullptr, nullptr);
		world.command_queue.enqueueWriteBuffer(world.buffer_Wc, CL_TRUE, 0, weights_in_bytes, Wc, nullptr, nullptr);
		world.command_queue.enqueueWriteBuffer(world.buffer_bc, CL_TRUE, 0, bias_in_bytes, bc, nullptr, nullptr);
		world.command_queue.enqueueWriteBuffer(world.buffer_Wo, CL_TRUE, 0, weights_in_bytes, Wo, nullptr, nullptr);
		world.command_queue.enqueueWriteBuffer(world.buffer_bo, CL_TRUE, 0, bias_in_bytes, bo, nullptr, nullptr);
		world.command_queue.enqueueWriteBuffer(world.buffer_W, CL_TRUE, 0, w_in_bytes, W, nullptr, nullptr);
		world.command_queue.enqueueWriteBuffer(world.buffer_b, CL_TRUE, 0, b_in_bytes, b, nullptr, nullptr);
	}
	//Launch the Kernel
	world.command_queue.enqueueTask(world.krnl);
	//Read output data from buffer
	world.command_queue.enqueueReadBuffer(world.buffer_res,CL_TRUE, 0, size_out_bytes, output, nullptr, nullptr);
}

void lstm_end()
{       
        clReleaseMemObject(world.buffer_data());
        clReleaseMemObject(world.buffer_Wi());
        clReleaseMemObject(world.buffer_bi());
        clReleaseMemObject(world.buffer_Wf());
        clReleaseMemObject(world.buffer_bf());
        clReleaseMemObject(world.buffer_Wc());
        clReleaseMemObject(world.buffer_bc());
        clReleaseMemObject(world.buffer_Wo());
        clReleaseMemObject(world.buffer_bo());
        clReleaseMemObject(world.buffer_W());
        clReleaseMemObject(world.buffer_b());
        clReleaseMemObject(world.buffer_res());
        world.command_queue.finish();
}

short int fixp(float value, int fracbits)
{
	int shiftv=pow(2.0,fracbits);
        int valfix=floor(value*shiftv);
        return (short int)valfix;
}

}
