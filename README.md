# CUDA 8-way Connected Component Labelling

An 8-way implementation of the Playne-equvalence algorithm for connected component labelling on CUDA.
Based on the illustrative example by Daniel Playne: <https://github.com/DanielPlayne/playne-equivalence-algorithm> as originally described in:

D. P. Playne and K. Hawick,
"A New Algorithm for Parallel Connected-Component Labelling on GPUs,"
in IEEE Transactions on Parallel and Distributed Systems,
vol. 29, no. 6, pp. 1217-1230, 1 June 2018.

* URL: <https://ieeexplore.ieee.org/document/8274991>

This was one part of a pipeline implemented for a GPU blob detection algorithm during my master's thesis for a degree in Master's of Science in engineering: Engineering Physics.

## Prerequisites

* `OpenCV` is used to load and display images, it is assumed that it has been installed correctly.
* `CUDA-toolkit`, This has been tested on an Nvidia Jetson TX2 running CUDA 9.0. Any newer version of the CUDA toolkit should be usable and many of the older ones as well. It does use managed memory, so your graphics card needs to be compatible with that. Per Nvidia the requirements are:

  * "a GPU with SM architecture 3.0 or higher (Kepler class or newer)"
  * "a 64-bit host application and non-embedded operating system (Linux, Windows, macOS)"

## Compiling

* Clone this repo onto your computer

* Edit the line `CUDAFLAGS = -arch=sm 62` in the makefile to whichever compute capability your graphics card uses. The info should be able to be found here: <https://developer.nvidia.com/cuda-gpus.>

* Run `make`

## Usage

`$ ./<main> <image-file>`

## License

The source code is provided under The MIT license