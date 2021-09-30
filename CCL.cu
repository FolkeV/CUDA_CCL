/* MIT License
 *
 * Copyright (c) 2018 - Daniel Peter Playne
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

#include "CCL.cuh"
#include "reduction.cuh"

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 4

/* Connected component labeling on binary images based on
 * the article by Playne and Hawick https://ieeexplore.ieee.org/document/8274991. */
void connectedComponentLabeling(unsigned int* outputImg, unsigned char* inputImg, size_t numCols, size_t numRows)
{
	// Create Grid/Block
	dim3 block (BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 grid ((numCols+BLOCK_SIZE_X-1)/BLOCK_SIZE_X,
			(numRows+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y);

	// Initialise labels
	init_labels<<< grid, block >>>(outputImg, inputImg, numCols, numRows);
	// Analysis
	resolve_labels <<< grid, block >>>(outputImg, numCols, numRows);

	// Label Reduction
	label_reduction <<< grid, block >>>(outputImg, inputImg, numCols, numRows);

	// Analysis
	resolve_labels <<< grid, block >>>(outputImg, numCols, numRows);

	// Force background to have label zero;
	resolve_background<<<grid, block>>>(outputImg, inputImg, numCols, numRows);
}

/* CUDA kernels
 */
__global__ void init_labels(unsigned int* g_labels, const unsigned char *g_image, const size_t numCols, const size_t numRows) {
	// Calculate index
	const unsigned int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int iy = (blockIdx.y * blockDim.y) + threadIdx.y;

	// Check Thread Range
	if((ix < numCols) && (iy < numRows)) {
		// Fetch five image values
		const unsigned char pyx = g_image[iy*numCols + ix];

		// Neighbour Connections
		const bool nym1x   =  (iy > 0) 					  	 ? (pyx == g_image[(iy-1) * numCols + ix  ]) : false;
		const bool nyxm1   =  (ix > 0)  		  			 ? (pyx == g_image[(iy  ) * numCols + ix-1]) : false;
		const bool nym1xm1 = ((iy > 0) && (ix > 0)) 		 ? (pyx == g_image[(iy-1) * numCols + ix-1]) : false;
		const bool nym1xp1 = ((iy > 0) && (ix < numCols -1)) ? (pyx == g_image[(iy-1) * numCols + ix+1]) : false;

		// Label
		unsigned int label;

		// Initialise Label
		// Label will be chosen in the following order:
		// NW > N > NE > E > current position
		label = (nyxm1)   ?  iy   *numCols + ix+1 : iy*numCols + ix;
		label = (nym1xp1) ? (iy-1)*numCols + ix+1 : label;
		label = (nym1x)   ? (iy-1)*numCols + ix   : label;
		label = (nym1xm1) ? (iy-1)*numCols + ix-1 : label;

		// Write to Global Memory
		g_labels[iy*numCols + ix] = label;
	}
}

// Resolve Kernel
__global__ void resolve_labels(unsigned int *g_labels,
		const size_t numCols, const size_t numRows) {
	// Calculate index
	const unsigned int id = ((blockIdx.y * blockDim.y) + threadIdx.y) * numCols +
							((blockIdx.x * blockDim.x) + threadIdx.x);

	// Check Thread Range
	if(id < (numRows* numCols)) {
		// Resolve Label
		g_labels[id] = find_root(g_labels, g_labels[id]);
	}
}

// Label Reduction
__global__ void label_reduction(unsigned int *g_labels, const unsigned char *g_image,
		const size_t numCols, const size_t numRows) {
	// Calculate index
	const unsigned int iy = ((blockIdx.y * blockDim.y) + threadIdx.y);
	const unsigned int ix = ((blockIdx.x * blockDim.x) + threadIdx.x);

	// Check Thread Range
	if((ix < numCols) && (iy < numRows)) {
		// Compare Image Values
		const unsigned char pyx = g_image[iy*numCols + ix];
		const bool nym1x = (iy > 0) ? (pyx == g_image[(iy-1)*numCols + ix]) : false;

		if(!nym1x) {
			// Neighbouring values
			const bool nym1xm1 = ((iy > 0) && (ix >  0)) 		 ? (pyx == g_image[(iy-1) * numCols + ix-1]) : false;
			const bool nyxm1   =              (ix >  0) 		 ? (pyx == g_image[(iy  ) * numCols + ix-1]) : false;
			const bool nym1xp1 = ((iy > 0) && (ix < numCols -1)) ? (pyx == g_image[(iy-1) * numCols + ix+1]) : false;

			if(nym1xp1){
				// Check Criticals
				// There are three cases that need a reduction
				if ((nym1xm1 && nyxm1) || (nym1xm1 && !nyxm1)){
					// Get labels
					unsigned int label1 = g_labels[(iy  )*numCols + ix  ];
					unsigned int label2 = g_labels[(iy-1)*numCols + ix+1];

					// Reduction
					reduction(g_labels, label1, label2);
				}

				if (!nym1xm1 && nyxm1){
					// Get labels
					unsigned int label1 = g_labels[(iy)*numCols + ix  ];
					unsigned int label2 = g_labels[(iy)*numCols + ix-1];

					// Reduction
					reduction(g_labels, label1, label2);
				}
			}
		}
	}
}

// Force background to get label zero;
__global__ void resolve_background(unsigned int *g_labels, const unsigned char *g_image,
		const size_t numCols, const size_t numRows){
	// Calculate index
	const unsigned int id = ((blockIdx.y * blockDim.y) + threadIdx.y) * numCols +
							((blockIdx.x * blockDim.x) + threadIdx.x);

	if(id < numRows*numCols){
		g_labels[id] = (g_image[id] > 0) ? g_labels[id]+1 : 0;
	}
}
