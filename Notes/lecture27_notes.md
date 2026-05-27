## DeepSeek Quantization
1. Mixed Precision Framework

<p align="center">
  <img src="https://github.com/muarshad01/DeepSeek/blob/main/images/lec26/deepseek-quantization.png" width="600" height="300" />
</p>

#### Fprop

|$y=x \times W$|
|---|
| $y_{FP32 \rightarrow BF16} = x_{BF16 \rightarrow FP8} \times W_{FP32 \rightarrow FP8}$ |

***

#### Dprop

||
|---|
| $z = x \times W$ |
| $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \times W^{T}$ |

$$
\begin{align}
   \bigg(\frac{\partial L}{\partial x}\bigg)_{FP32} &= \bigg(\frac{\partial L}{\partial z}\bigg)_{FP8} \times W^{T}_{FP8}\\
   \bigg(\frac{\partial L}{\partial x}\bigg)_{FP32 \rightarrow BF16} &= \bigg(\frac{\partial L}{\partial z}\bigg)_{ BF16 \rightarrow FP8} \times W^{T}_{ FP32 \rightarrow FP8}
\end{align}
$$

***

#### Xprop

$$
\begin{align}
   \bigg(\frac{\partial L}{\partial W}\bigg)_{FP32} &= x^{T} \times \bigg(\frac{\partial L}{\partial z}\bigg)_{FP8} \\
   \bigg(\frac{\partial L}{\partial x}\bigg)_{FP32 \rightarrow BF16} &= \bigg(\frac{\partial L}{\partial z}\bigg)_{ BF16 \rightarrow FP8} \times W^{T}_{ FP32 \rightarrow FP8}
\end{align}
$$

***

2. Fine-grained quantization

***


#### Fine-Grained Quantization
* Seperate scaling
* GEMM (General Matrix Multiply)

***
