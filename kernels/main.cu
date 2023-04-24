#include "main.cuh"

__host__
__device__
inline int get_cube_index(int width, int height, int planeNb, int x, int y)
{
	return (planeNb * width * height) + (y * width) + x;
}

__global__
void dev_sweep(gpuCam* cams, float* result, int window, int nbCams, int ref_cam_index)
{
	int threadGlobalId = blockDim.x * blockIdx.x + threadIdx.x;
	gpuCam ref = cams[ref_cam_index];

	if (threadGlobalId >= ref.width * ref.height)
	{
		return;
	}

	// Mapping thread ID to 2D coordinate on the plane
	int x = threadGlobalId % ref.width;
	int y = threadGlobalId / ref.width;

	// For each camera, calculating cost cube value for pixel
	for (int i = 0; i < nbCams; i++)
	{
		// Skipping reference camera
		if (i == ref_cam_index)
		{
			continue;
		}

		// Treating all the layers for the pixel
		for (int zi = 0; zi < ZPlanes; zi++)
		{
			double z = ZNear * ZFar / (ZNear + (((double)zi / (double)ZPlanes) * (ZFar - ZNear)));

			double X_ref = (ref.K_inv[0] * x + ref.K_inv[1] * y + ref.K_inv[2]) * z;
			double Y_ref = (ref.K_inv[3] * x + ref.K_inv[4] * y + ref.K_inv[5]) * z;
			double Z_ref = (ref.K_inv[6] * x + ref.K_inv[7] * y + ref.K_inv[8]) * z;

			double X = ref.R_inv[0] * X_ref + ref.R_inv[1] * Y_ref + ref.R_inv[2] * Z_ref - ref.t_inv[0];
			double Y = ref.R_inv[3] * X_ref + ref.R_inv[4] * Y_ref + ref.R_inv[5] * Z_ref - ref.t_inv[1];
			double Z = ref.R_inv[6] * X_ref + ref.R_inv[7] * Y_ref + ref.R_inv[8] * Z_ref - ref.t_inv[2];

			double X_proj = cams[i].R[0] * X + cams[i].R[1] * Y + cams[i].R[2] * Z - cams[i].t[0];
			double Y_proj = cams[i].R[3] * X + cams[i].R[4] * Y + cams[i].R[5] * Z - cams[i].t[1];
			double Z_proj = cams[i].R[6] * X + cams[i].R[7] * Y + cams[i].R[8] * Z - cams[i].t[2];

			double x_proj = (cams[i].K[0] * X_proj / Z_proj + cams[i].K[1] * Y_proj / Z_proj + cams[i].K[2]);
			double y_proj = (cams[i].K[3] * X_proj / Z_proj + cams[i].K[4] * Y_proj / Z_proj + cams[i].K[5]);
			double z_proj = Z_proj;

			x_proj = x_proj < 0 || x_proj >= cams[i].width ? 0 : roundf(x_proj);
			y_proj = y_proj < 0 || y_proj >= cams[i].height ? 0 : roundf(y_proj);

			float cost = 0.0f;
			float cc = 0.0f;

			for (int k = -window / 2; k <= window / 2; k++)
			{
				for (int l = -window / 2; l <= window / 2; l++)
				{
					if (x + l < 0 || x + l >= ref.width)
						continue;
					if (y + k < 0 || y + k >= ref.height)
						continue;
					if (x_proj + l < 0 || x_proj + l >= cams[i].width)
						continue;
					if (y_proj + k < 0 || y_proj + k >= cams[i].height)
						continue;

					uint8_t refCamYChannel = ref.YChannelData[((y + k) * ref.width) + (x + l)];
					uint8_t currentCamYChannel = cams[i].YChannelData[(((int)y_proj + k) * cams[i].width) + ((int)x_proj + l)];

					cost += fabs((double)(refCamYChannel - currentCamYChannel));

					cc += 1.0f;
				}
			}

			cost /= cc;

			int resultIndex = get_cube_index(ref.width, ref.height, zi, x, y);
			result[resultIndex] = fminf(result[resultIndex], cost);
		}
	}
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

std::vector<cv::Mat> gpu_sweeping_plane(std::vector<cam> const& cam_vector, int ref_cam_index, int window)
{
	// 1 - Preprocess data and set GPU device
	cudaSetDevice(0);

	gpuCam* gpuCams = transform_cams(cam_vector);
	const gpuCam ref = gpuCams[ref_cam_index];

	int resultArrSize = ZPlanes * ref.width * ref.height;
	float* res = (float*)malloc(resultArrSize * sizeof(float));

	for (int i = 0; i < resultArrSize; i++)
	{
		res[i] = 255.f;
	}

	gpuCam* dev_gpuCameras;
	float* dev_result;

	cudaMalloc((void**)&dev_gpuCameras, sizeof(gpuCam) * cam_vector.size());
	cudaMalloc((void**)&dev_result, resultArrSize * sizeof(float));

	// 2 - Copy data to GPU
	cudaMemcpy(dev_gpuCameras, gpuCams, sizeof(gpuCam) * cam_vector.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_result, res, resultArrSize * sizeof(float), cudaMemcpyHostToDevice);

	// 3 - Launch kernel and wait for it to finish
	const dim3 nbBlocks(NB_BLOCKS);
	const dim3 nbThreadsPerBlock(NB_THREADS_PER_BLOCK);

	cout << "Launching GPU kernel on " << NB_BLOCKS << " blocks and " << NB_THREADS_PER_BLOCK << " threads per block" << endl;

	dev_sweep << <nbBlocks, nbThreadsPerBlock >> > (dev_gpuCameras, dev_result, window, cam_vector.size(), ref_cam_index);

	cudaDeviceSynchronize();

	cout << "Kernel exited" << endl;

	auto err = cudaGetLastError();
	printf("CUDA status : %s\n\n", cudaGetErrorString(err));

	// 4 - Extract result from GPU
	cudaMemcpy(res, dev_result, resultArrSize * sizeof(float), cudaMemcpyDeviceToHost);

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

	cout << "GPU memory has been freed" << endl;
	cout << "Re-building opencv wrapper (OpenCV::Mat) cost cube vector" << endl;

	std::vector<cv::Mat> cost_cube(ZPlanes);

	for (int i = 0; i < cost_cube.size(); i++)
	{
		cost_cube[i] = cv::Mat(ref.height, ref.width, CV_32FC1, 255.);
		
		for (int j = 0; j < ref.height; j++) {
			for (int k = 0; k < ref.width; k++) {
				int res_pos = (i * ref.width * ref.height) + (j * ref.width) + k;
				cost_cube[i].at<float>(j, k) = res[res_pos];
			}
		}
	}

	cout << "Cost cube vector has been re-built" << endl << endl;

	free(res);

	cudaDeviceReset();

	return cost_cube;
}