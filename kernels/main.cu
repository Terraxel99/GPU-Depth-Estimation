#include "main.cuh"

__global__ void dev_sweep(gpuCam* cam, char* x)
{
	*x = cam[0].name[2];
}

gpuCam* transform_cams(vector<cam> const& cam_vector)
{
	gpuCam* cameras = (gpuCam*)malloc(sizeof(gpuCam) * cam_vector.size());

	for (int i = 0; i < cam_vector.size(); i++)
	{
		cam camera = cam_vector.at(i);
		char* name;

		cudaMalloc((void**)&name, camera.name.size());
		cudaMemcpy(name, camera.name.data(), camera.name.size(), cudaMemcpyHostToDevice);

		cameras[i].size = camera.size;
		cameras[i].width = camera.width;
		cameras[i].height = camera.height;
		cameras[i].name = name;
	}

	return cameras;
}

void gpu_sweeping_plane(cam const ref, std::vector<cam> const& cam_vector, int window)
{
	// 1 - Preprocess data and set GPU device
	cudaSetDevice(0);

	gpuCam* gpuCams = transform_cams(cam_vector);
	gpuCam* dev_gpuCameras;
	char* dev_result;

	cudaMalloc((void**)&dev_gpuCameras, sizeof(gpuCam) * cam_vector.size());
	cudaMalloc((void**)&dev_result, sizeof(char));

	// 2 - Copy data to GPU
	cudaMemcpy(dev_gpuCameras, gpuCams, sizeof(gpuCam) * cam_vector.size(), cudaMemcpyHostToDevice);

	// 3 - Launch kernel and wait for it to finish
	const dim3 nbBlocks(1);
	const dim3 nbThreadsPerBlock(1);

	dev_sweep << <nbBlocks, nbThreadsPerBlock >> > (dev_gpuCameras, dev_result);
	cudaDeviceSynchronize();
	
	// 4 - Extract result from GPU
	char res = 0;
	cudaMemcpy(&res, dev_result, sizeof(char), cudaMemcpyDeviceToHost);

	cout << res << endl;
	cout << cam_vector.at(0).name.at(2) << endl;
	
	// 5 - Free CPU and/or GPU memory
	for (int i = 0; i < cam_vector.size(); i++)
	{
		cudaFree(&(dev_gpuCameras[i].name));
		cudaFree(&(gpuCams[i].name));
	}

	cudaFree(dev_gpuCameras);
	cudaFree(gpuCams);

	cudaFree(dev_result);
	
	// 6 - Build & return cost cube	(TODO)
}
