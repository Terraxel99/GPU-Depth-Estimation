#pragma once

typedef struct gpuCam {
	int width;
	int height;
	int size;
	char* name;
	uint8_t* YChannelData;
	double* K;
	double* R;
	double* t;
	double* K_inv;
	double* R_inv;
	double* t_inv;
} gpuCam;
