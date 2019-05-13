/*
 * main.cu
 *
 *  Created on: May 8, 2019
 *      Author: Folke Vesterlund
 */

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "CCL.cuh"
#include "utils.hpp"

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

	// Run kernel
	connectedComponentLabeling(d_labels, d_img, numCols, numRows);
	cudaDeviceSynchronize();

	// Plot result
	cv::Mat finalImage = util::postProc(d_labels, numCols, numRows);
	cv::imshow("Labelled image", finalImage);
	cv::waitKey();

	// Free memory
	cudaFree(d_img);
	cudaFree(d_labels);
}
