# src-hasp
This repository contains the source files used to run the resnet benchmark from the MLPerf tiny benchmark suite using HASP (or HHT) architecutre on our modified version of the RISC-V Spike simulator.

Each header file contains the code for a separate DNN layer as well as utility functions for 8-bit quantization of input/output values and compressiong sparse tensors into PSR format. 

There are mulitple implementations of the convolution layer to test the performance of the hasp architecture against a single-core architecture. hht_func.cc shows the portion of the convolution algorithm that is performed in hardware by the memory-side accelerator as outlined in https://csrl.cse.unt.edu/kavi/Research/SBAC-PAD-2022.pdf.

More details can be found in the paper.
