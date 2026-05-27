## DeepSeek Quantization
1. Mixed Precision Framework
2. Fine-grained quantization

***

||
|---|
| $x_{BF16 \rightarrow FP8}$ |
| $W_{ \rightarrow FP8}$ |
| $y_{FP32 \rightarrow BF16}$ |
| $y_{FP32 \rightarrow BF16} = x_{BF16 \rightarrow FP8} \times W_{ \rightarrow FP8}$ |

***



#### Fine-Grained Quantization
* Seperate scaling
* GEMM (General Matrix Multiply)

***
