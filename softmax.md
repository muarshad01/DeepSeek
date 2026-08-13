## Softmax

$$\text{Attention}(Q,K,V)=\text{softmax}\bigg(\frac{QK^T}{\sqrt{d_{k}}}\bigg)V$$

#### Example
$$\\{x_1, x_2, x_3, x_4, x_5, x_6\\}$$

$$\text{sum} = e^{x_1} + e^{x_2} + e^{x_3} + e^{x_4} + e^{x_5} + e^{x_6}$$

$$\bigg\\{\frac{e^{x_1}}{\text{sum}},\frac{e^{x_2}}{\text{sum}},\frac{e^{x_3}}{\text{sum}},\frac{e^{x_4}}{\text{sum}},\frac{e^{x_5}}{\text{sum}},\frac{e^{x_6}}{\text{sum}},\bigg\\}$$

* __Peaky Output__: Softmax gives more attention to higher values and less attention to lower values.
* ALL values sum up to 1.

#### Scaling
* Why divide by $\sqrt{d_{keys}}$
* Variance scales as $\sqrt{d_{keys}}$

***

