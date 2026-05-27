#### Increasing accumulation precision
* When performing low-precision GEMM (General Matrix Multiplication) operations, there are two major issues:
1. __Low-precision underflow__: With limited precision (FP8), you quickly loose accuracy due to small intermediate results becoming too small ("underflow") or precision limitations during accumulation.
2. __Limited accumulation precision__: NVIDIA tensor cores (such as H800 on the GPU) accumulate GEMM results internally with limited precision (~14 bits), far below standard FP32 accumulation precision

***

#### Mantissa over Exponents

***

#### Online Quantization

***
