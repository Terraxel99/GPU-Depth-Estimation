#pragma once

#include <cuda_runtime.h>
#include <cuda.h>

#include <iostream>
#include <cstdio>

#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/highgui/highgui.hpp>

#include "../src/constants.hpp"
#include "gpu_cam.hpp"

#define NB_BLOCKS 16200
#define NB_THREADS_PER_BLOCK 128
#define NB_ELEMENT_PER_THREAD 1

using namespace std;

// This is the public interface of our cuda function, called directly in main.cpp
std::vector<cv::Mat> gpu_sweeping_plane(std::vector<cam> const& cam_vector, int ref_cam_index, int window, float* runtime);