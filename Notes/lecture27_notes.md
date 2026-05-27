## DeepSeek Quantization
1. Mixed Precision Framework

<p align="center">
  <img src="https://github.com/muarshad01/DeepSeek/blob/main/images/lec26/deepseek-quantization.png" width="600" height="300" />
</p>

***

#### Fprop

$$
\begin{align}
   y_{BF16}                  &= x_{FP8} \times W_{FP8}\\
   y_{FP32 \rightarrow BF16} &= x_{BF16 \rightarrow FP8} \times W_{FP32 \rightarrow FP8}
\end{align}
$$
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

#### Wprop

$$
\begin{align}
   \bigg(\frac{\partial L}{\partial W}\bigg)_{FP32} &= x^{T} \times \bigg(\frac{\partial L}{\partial z}\bigg)_{FP8} \\
   \bigg(\frac{\partial L}{\partial W}\bigg)_{FP32} &= x^{T}_{FP8} \times \bigg(\frac{\partial L}{\partial z}\bigg)_{ BF16 \rightarrow FP8} 
\end{align}
$$

* After updating, master weights are converted to BF16 for next iteration forward pass or FP8 conversions when required.

***

* 15:00

#### Fine-Grained Quantization
* Seperate scaling
* GEMM (General Matrix Multiply)

***
