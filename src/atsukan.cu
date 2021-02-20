#include <iostream>
#include <functional>
#include <chrono>
#include <thread>
#include <cutf/memory.hpp>
#include <cutf/nvrtc.hpp>
#include <cuda_kernel_fusing.hpp>

std::string gen_fma_kernel(const unsigned n_fma, const unsigned n_inner_loop) {

	const auto loop_sentence = "for (unsigned i = 0; i < " + std::to_string(n_inner_loop) + R"(; i++) { float a = src_ptr[(tid + i) % array_size];)";

	cuda_kernel_fusing::kernel_constructor kernel_constructor(
			"float* const dst_ptr, const float* const src_ptr, size_t array_size",
			"const unsigned tid, float& a",
			"const unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;" + loop_sentence,
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
		"-arch=sm_86",
	};
	nvrtcResult result = nvrtcCompileProgram(program, 1, options);
	size_t log_size;
	nvrtcGetProgramLogSize(program,&log_size);
	char *log = new char[log_size];
	nvrtcGetProgramLog(program,log);
	//std::printf("// ------------- COMPILATION LOG ------------\n%s\n", log);
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
	//std::printf("// ------------- PTX KERNEL ------------\n%s\n", ptx);

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
	auto tmp_array_size = array_size;

	const auto block_size = 256;
	const auto grid_size = array_size / block_size;

	const auto start_clock = std::chrono::high_resolution_clock::now();
	for (std::size_t i = 0; i < n_execution; i++) {
		cutf::nvrtc::launch_function(
				cuFunction,
				{reinterpret_cast<void*>(&src_ptr), reinterpret_cast<void*>(&dst_ptr), reinterpret_cast<void*>(&tmp_array_size)},
				grid_size,
				block_size
				);
	}

	CUTF_CHECK_ERROR(cudaDeviceSynchronize());
	const auto end_clock = std::chrono::high_resolution_clock::now();
	const auto elapsed_time_per_kernel = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1e-6 / n_execution;
	std::printf("[INFO] %20s : %e[s]\n", "time per kernel", elapsed_time_per_kernel);
}

int main() {
	const auto start_clock = std::chrono::high_resolution_clock::now();
	for (unsigned n_op_log = 1; n_op_log < 8; n_op_log++) {
		for (unsigned n_inner_loop_log = 1; n_inner_loop_log < 8; n_inner_loop_log++) {
			const unsigned n_op = 1u << n_op_log;
			const unsigned n_inner_loop = 1u << n_inner_loop_log;

			const auto end_clock = std::chrono::high_resolution_clock::now();
			const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1e-6;
			std::printf("[INFO] %20s : time = %e [s], n_op = %5u, n_inner_loop = %5u\n", "start kernel", elapsed_time, n_op, n_inner_loop);

			const std::size_t n_all_op = 1lu << 17;
			const auto n_execution = n_all_op / (n_op * n_inner_loop);
			run_kernel(n_op, n_inner_loop, 1lu << 30, gen_fma_kernel, n_execution);

			std::this_thread::sleep_for(std::chrono::seconds(4));
		}
	}
}
