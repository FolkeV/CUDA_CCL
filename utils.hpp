/*
 * utils.hpp
 *
 *  Created on: May 8, 2019
 *      Author: Folke Vesterlund
 */

#ifndef UTILS_HPP_
#define UTILS_HPP_

namespace util{
	int mean(const unsigned char* img, const int N){
		int mean = 0;

		for(int i = 0; i<N; i++){
			mean += img[i];
		}
		mean /= N;

		return mean;
	}

	void threshold(unsigned char* outputImg, unsigned char* inputImg, size_t mean, size_t N){
		for (int i = 0; i < N; i++){
			outputImg[i] = inputImg[i] < mean ? 255:0;
		}
	}

	cv::Mat postProc(unsigned int* img, size_t numCols, size_t numRows){
		// Initilise a Mat to the correct size and all zeros
		cv::Mat outputImg(numRows, numCols, CV_8UC1, cv::Scalar::all(0));

		for (int i = 0; i < numRows; i++){
			for(int j = 0; j< numCols; j++){
				size_t idx = i * numCols + j;
				if (img[idx] > 0){
					// The background will 0, force all labels a bit away from zero
					// to be able to visualise better
					outputImg.at<uchar>(i,j) = img[idx]%240+15;
				}
			}
		}

		applyColorMap(outputImg, outputImg, cv::COLORMAP_JET);

		return outputImg;
	}
}

#endif /* UTILS_HPP_ */
