#### Increasing accumulation precision
* When performing low-precision GEMM (General Matrix Multiplication) operations, there are two major issues:
1. __Low-precision underflow__: With limited precision (FP8), you quickly loose accuracy due to small intermediate results becoming too small ("underflow") or precision limitations during accumulation.
2. __Limited accumulation precision__: NVIDIA tensor cores (such as H800 on the GPU) accumulate GEMM results internally with limited precision (~14 bits), far below standard FP32 accumulation precision

* When performing GEMM operations with large inner dimensions K, low accumulation precision can lead to significant numerical errors.
* Multiplying matrices with inner dimension k=4096, low accumulation precision can cause a relative error as large as ~2%, which heavily impacts model accuracy.

#### DeepSeek Solution: Promotion to CUDA cores
* The authors propose temporatily moving intermediate accumulation results from low-precision Tensor-cores to high-precision CUDA-cores (FP32) periodically during computation. This technique is called "promotion to CUDA cores"

* **Step-1**: Low-precision MMA (Tensor core)
  * Initially, MMA operations are performed using FP8 precision on Tensor cores.
  * The intermediate results (Low Prec Acc) accumulate internally with limited precision (~14 bits).
  * Warp Group Level Matrix Multiply Accumulate (WGMMA)

* **Step-2**: Promotion to higher-precision (CUDA core)
  * After a certain interval (denoted NC, typically 128 elements), the partial low-precision accumulations are promoted (copied) to high-precision (FP32) registers in CUDA cores.
  * Scaling factors from fine-grained quantization are multiplied during dequantization.

***

* 11:00

#### Mantissa over Exponents
* In FP8, the number of bits assigned to Exponent (dynamic range) and Mantissa (precision) heavily influence numerical precision and representable range.

| Format | Exponent bits (Dynamic Range) | Mantissa bits (Precision) |
|---|---|---|
| E4M3 | 4-bits (smaller range) | 3-bits (higher precision) |
| E5M2 | 5-bits (larger range)  | 2-bits (lower precision)  |


***

#### Online Quantization

***
