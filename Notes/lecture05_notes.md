#### Scaled Dot-Product Attention

$$\text{Attention}(Q,K,V)=\text{softmax}\bigg(\frac{QK^T}{\sqrt{d_{k}}}\bigg)V$$


#### Softmax

$$\{x_1, x_2, x_3, x_4, x_5, x_6\}$$

$$\{\frac{e^{x_1}}{\text{sum}},\frac{e^{x_2}}{\text{sum}},\frac{e^{x_3}}{\text{sum}},\frac{e^{x_4}}{\text{sum}},\frac{e^{x_5}}{\text{sum}},\frac{e^{x_6}}{\text{sum}},\}$$

$$\text{sum} = e^{x_2} + e^{x_2} + e^{x_3} + e^{x_4} + e^{x_5} + e^{x_6}$$

***





