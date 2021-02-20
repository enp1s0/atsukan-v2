#include <iostream>
#include <functional>
#include <cutf/memory.hpp>
#include <cutf/nvrtc.hpp>
#include <cuda_kernel_fusing.hpp>

std::string gen_fma_kernel(const unsigned n_fma, const unsigned n_inner_loop) {

	const auto loop_sentence = "for (unsigned i = 0; i < " + std::to_string(n_inner_loop) + "; i++) {";

	cuda_kernel_fusing::kernel_constructor kernel_constructor(
			"float* const dst_ptr, const float* const src_ptr",
			"const unsigned tid, float& a",
			"const unsigned tid = threadIdx.x + blockIdx.x * blockDim.x; float a = src_ptr[tid];" + loop_sentence,
			"dst_ptr[tid] = a;}"
			);
	kernel_constructor.add_device_function(
			"fma_unit",
			R"(
{
	//a = fmaf(a, a, a);
	a = a * a + a;
}
)"
			);
	std::vector<std::string> func_list(n_fma);
	for (auto& f : func_list) f = "fma_unit";

	return kernel_constructor.generate_kernel_code(func_list);
}

void run_kernel(const unsigned n_op, const unsigned n_inner_loop, const std::size_t array_size, const std::function<std::string(const unsigned, const unsigned)> gen_kernel_func, const std::size_t n_execution) {
	auto src_mem = cutf::memory::get_device_unique_ptr<float>(array_size);
	auto dst_mem = cutf::memory::get_device_unique_ptr<float>(array_size);

	const auto kernel_code = gen_kernel_func(n_op, n_inner_loop);
	std::printf("// ------------- CU KERNEL ------------\n%s\n", kernel_code.c_str());

	nvrtcProgram program;
	nvrtcCreateProgram(&program,
			kernel_code.c_str(),
			"kernel.cu",
			0,
			NULL,
			NULL);
	const char *options[] = {
		"--gpu-architecture=compute_75",
	};
	nvrtcResult result = nvrtcCompileProgram(program, 1, options);
	size_t log_size;
	nvrtcGetProgramLogSize(program,&log_size);
	char *log = new char[log_size];
	nvrtcGetProgramLog(program,log);
	std::printf("// ------------- COMPILATION LOG ------------\n%s\n", log);
	delete [] log;
	if(result != NVRTC_SUCCESS){
		std::cerr<<"Compilation failed"<<std::endl;
		return;
	}


	// Get PTX
	std::size_t ptx_size;
	nvrtcGetPTXSize(program, &ptx_size);
	char *ptx = new char [ptx_size];
	nvrtcGetPTX(program, ptx);
	nvrtcDestroyProgram(&program);
	std::printf("// ------------- PTX KERNEL ------------\n%s\n", ptx);

	// Create kernel image
	CUdevice cuDevice;
	CUcontext cuContext;
	CUmodule cuModule;
	CUfunction cuFunction;
	cuInit(0);
	cuDeviceGet(&cuDevice, 0);
	cuCtxCreate(&cuContext, 0, cuDevice);
	cuModuleLoadDataEx(&cuModule, ptx, 0, 0, 0);
	cuModuleGetFunction(&cuFunction, cuModule, "cukf_main");
	delete [] ptx;


	auto src_ptr = src_mem.get();
	auto dst_ptr = dst_mem.get();

	const auto block_size = 256;
	const auto grid_size = array_size / block_size;

	for (std::size_t i = 0; i < n_execution; i++) {
		cutf::nvrtc::launch_function(
				cuFunction,
				{&src_ptr, &dst_ptr},
				grid_size,
				block_size
				);
	}

	CUTF_CHECK_ERROR(cudaDeviceSynchronize());
}

int main() {
	run_kernel(4, 16, 1lu << 30, gen_fma_kernel, 1lu << 15);
}
