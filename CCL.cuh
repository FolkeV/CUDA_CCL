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

#ifndef CCL_CUH_
#define CCL_CUH_

void connectedComponentLabeling(unsigned int* outputImg, unsigned char* inputImg, size_t numCols, size_t numRows);

/* CUDA kernels
 */

// Initialise Kernel
__global__ void init_labels(unsigned int* g_labels, const unsigned char *g_image,
		const size_t numCols, const size_t numRows);

// Resolve Kernel
__global__ void resolve_labels(unsigned int *g_labels,
		const size_t numCols, const size_t numRows);

// Label Reduction
__global__ void label_reduction(unsigned int *g_labels, const unsigned char *g_image,
		const size_t numCols, const size_t numRows);

// Force background to have label '0'.
__global__ void resolve_background(unsigned int *g_labels, const unsigned char *g_image,
		const size_t numCols, const size_t numRows);



#endif /* CCL_CUH_ */


