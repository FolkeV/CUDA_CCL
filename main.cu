/* MIT License
 *
 * Copyright (c) 2019 - Folke Vesterlund
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "CCL.cuh"
#include "utils.hpp"
#include "timer.h"

int main(int argc,char **argv){
	std::string fileName;
	size_t numPixels, numRows, numCols;

	if (argc < 2){
		std::cout << "Usage: "<< argv[0] << " <image file>" << std::endl;
		return(-1);
	}
	fileName = argv[1];

	// Read image
	cv::Mat image;
	image = cv::imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);
	if(!image.data){
		std::cerr << "Couldn't open file" << std::endl;
		return(-1);
	}

	if(!image.isContinuous()){
		std::cerr << "Image is not allocated with continuous data. Exiting..." << std::endl;
		return(-1);
	}
	numCols = image.cols;
	numRows = image.rows;
	numPixels = numRows*numCols;

	// Allocate GPU data
	// Uses managed data, so no explicit copies are needed
	unsigned char* d_img;
	unsigned  int* d_labels;
	cudaMallocManaged(&d_labels, numPixels * sizeof(int ));
	cudaMallocManaged(&d_img   , numPixels * sizeof(char));

	// Pre process image
	int imgMean = util::mean(image.data, numPixels);
	util::threshold(d_img, image.data, imgMean, numPixels);

	// Run and time kernel
	GpuTimer timer;
	timer.Start();
	connectedComponentLabeling(d_labels, d_img, numCols, numRows);
	timer.Stop();
	std::cout << "GPU code ran in: " << timer.Elapsed() << "ms" << std::endl;
//	cudaDeviceSynchronize();	// Timer has syncronization built in
	
	// Count components
	unsigned int components = util::countComponents(d_labels, numPixels);
	std::cout << "Number of components: " << components << std::endl;

	// Plot result
	cv::Mat finalImage = util::postProc(d_labels, numCols, numRows);
	cv::imshow("Labelled image", finalImage);
	cv::waitKey();

	// Free memory
	cudaFree(d_img);
	cudaFree(d_labels);
}
