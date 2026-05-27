## DeepSeek Quantization
1. Mixed Precision Framework

<p align="center">
  <img src="https://github.com/muarshad01/DeepSeek/blob/main/images/lec26/deepseek-quantization.png" width="600" height="300" />
</p>

|$y=x \times W$|
|---|
| $y_{FP32 \rightarrow BF16} = x_{BF16 \rightarrow FP8} \times W_{FP32 \rightarrow FP8}$ |

***

2. Fine-grained quantization

***


#### Fine-Grained Quantization
* Seperate scaling
* GEMM (General Matrix Multiply)

***
