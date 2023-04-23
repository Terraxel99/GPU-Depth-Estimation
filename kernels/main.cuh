#pragma once

#include <cuda_runtime.h>
#include <cuda.h>

#include <iostream>
#include <cstdio>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../src/constants.hpp"
#include "gpu_cam.hpp"

using namespace std;

// This is the public interface of our cuda function, called directly in main.cpp
void gpu_sweeping_plane(std::vector<cam> const& cam_vector, int ref_cam_index, int window);