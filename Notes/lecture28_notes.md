#### Increasing accumulation precision
* When performing low-precision GEMM (General Matrix Multiplication) operations, there are two major issues:
1. __Low-precision underflow__: With limited precision (FP8), you quickly loose accuracy due to small intermediate results becoming too small ("underflow") or precision limitations during accumulation.
2. __Limited accumulation precision__: NVIDIA tensor cores (such as H800 on the GPU) accumulate GEMM results internally with limited precision (~14 bits), far below standard FP32 accumulation precision

* When performing GEMM operations with large inner dimensions K, low accumulation precision can lead to significant numerical errors.
* Multiplying matrices with inner dimension k=4096, low accumulation precision can cause a relative error as large as ~2%, which heavily impacts model accuracy.

#### DeepSeek Solution: Promotion to CUDA cores
* The authors propose temporatily moving intermediate accumulation results from low-precision Tensor-cores to high-precision CUDA-cores (FP32) periodically during computation. This technique is called "promotion to CUDA cores"

***

#### Mantissa over Exponents

***

#### Online Quantization

***
