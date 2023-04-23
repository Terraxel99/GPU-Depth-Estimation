#include "main.cuh"

__global__ void dev_sweep(gpuCam* cam, int ref_cam_index, float* result)
{
	result[2084] = (float)ref_cam_index + 168;
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

		double* K;
		cudaMalloc((void**)&K, camera.p.K.size() * sizeof(double));
		cudaMemcpy(K, camera.p.K.data(), camera.p.K.size() * sizeof(double), cudaMemcpyHostToDevice);

		double* R;
		cudaMalloc((void**)&R, camera.p.R.size() * sizeof(double));
		cudaMemcpy(R, camera.p.R.data(), camera.p.R.size() * sizeof(double), cudaMemcpyHostToDevice);

		double* t;
		cudaMalloc((void**)&t, camera.p.t.size() * sizeof(double));
		cudaMemcpy(t, camera.p.t.data(), camera.p.t.size() * sizeof(double), cudaMemcpyHostToDevice);

		double* K_inv;
		cudaMalloc((void**)&K_inv, camera.p.K_inv.size() * sizeof(double));
		cudaMemcpy(K_inv, camera.p.K_inv.data(), camera.p.K_inv.size() * sizeof(double), cudaMemcpyHostToDevice);

		double* R_inv;
		cudaMalloc((void**)&R_inv, camera.p.R_inv.size() * sizeof(double));
		cudaMemcpy(R_inv, camera.p.R_inv.data(), camera.p.R_inv.size() * sizeof(double), cudaMemcpyHostToDevice);

		double* t_inv;
		cudaMalloc((void**)&t_inv, camera.p.t_inv.size() * sizeof(double));
		cudaMemcpy(t_inv, camera.p.t_inv.data(), camera.p.t_inv.size() * sizeof(double), cudaMemcpyHostToDevice);

		uint8_t* YChannelData;
		cudaMalloc((void**)&YChannelData, camera.YUV[0].total() * camera.YUV[0].elemSize());
		cudaMemcpy(YChannelData, camera.YUV[0].data, camera.YUV[0].total() * camera.YUV[0].elemSize(), cudaMemcpyHostToDevice);

		cameras[i].size = camera.size;
		cameras[i].width = camera.width;
		cameras[i].height = camera.height;
		cameras[i].name = name;
		cameras[i].K = K;
		cameras[i].R = R;
		cameras[i].t = t;
		cameras[i].K_inv = K_inv;
		cameras[i].R_inv = R_inv;
		cameras[i].t_inv = t_inv;
		cameras[i].YChannelData = YChannelData;
	}

	return cameras;
}

void gpu_sweeping_plane(std::vector<cam> const& cam_vector, int ref_cam_index, int window)
{
	// 1 - Preprocess data and set GPU device
	cudaSetDevice(0);

	gpuCam* gpuCams = transform_cams(cam_vector);
	gpuCam* dev_gpuCameras;
	float* dev_result;

	cudaMalloc((void**)&dev_gpuCameras, sizeof(gpuCam) * cam_vector.size());

	int resSize = ZPlanes * cam_vector.at(0).width * cam_vector.at(0).height * sizeof(float);
	cudaMalloc((void**)&dev_result, resSize);

	// 2 - Copy data to GPU
	cudaMemcpy(dev_gpuCameras, gpuCams, sizeof(gpuCam) * cam_vector.size(), cudaMemcpyHostToDevice);

	// 3 - Launch kernel and wait for it to finish
	const dim3 nbBlocks(1);
	const dim3 nbThreadsPerBlock(1);

	dev_sweep << <nbBlocks, nbThreadsPerBlock >> > (dev_gpuCameras, ref_cam_index, dev_result);
	cudaDeviceSynchronize();

	// 4 - Extract result from GPU
	float* res = (float*)malloc(resSize);
	cudaMemcpy(res, dev_result, resSize, cudaMemcpyDeviceToHost);

	// 5 - Free CPU and/or GPU memory
	for (int i = 0; i < cam_vector.size(); i++)
	{
		cudaFree(&(dev_gpuCameras[i].name));
		cudaFree(&(gpuCams[i].name));
		cudaFree(&(dev_gpuCameras[i].K));
		cudaFree(&(gpuCams[i].K));
		cudaFree(&(dev_gpuCameras[i].R));
		cudaFree(&(gpuCams[i].R));
		cudaFree(&(dev_gpuCameras[i].t));
		cudaFree(&(gpuCams[i].t));
		cudaFree(&(dev_gpuCameras[i].K_inv));
		cudaFree(&(gpuCams[i].K_inv));
		cudaFree(&(dev_gpuCameras[i].R_inv));
		cudaFree(&(gpuCams[i].R_inv));
		cudaFree(&(dev_gpuCameras[i].t_inv));
		cudaFree(&(gpuCams[i].t_inv));
		cudaFree(&(dev_gpuCameras[i].YChannelData));
		cudaFree(&(gpuCams[i].YChannelData));
	}

	cudaFree(dev_gpuCameras);
	cudaFree(gpuCams);

	cudaFree(dev_result);

	// 6 - Build & return cost cube	(TODO) + free its memory space

	free(res);
}