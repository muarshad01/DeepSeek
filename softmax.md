## Softmax

$$
\begin{align}
   S &= \text{casual mask}\bigg(\frac{Q \times K^T}{\sqrt{d_{keys}}}\bigg)\\
   P &= \text{softmax}\bigg(\text{casual mask}\bigg(\frac{Q \times K^T}{\sqrt{d_{keys}}}\bigg)\bigg)\\
   \text{Attention}(Q,K,V) = O &= \text{softmax}\bigg(\text{casual mask}\bigg(\frac{Q \times K^T}\\{\sqrt{d_{keys}}}\bigg)\bigg) \times V\\
\end{align}
$$

#### Example: Normal Softmax
$$\\{x_1, x_2, x_3, x_4, x_5, x_6\\}$$

$$\text{sum} = e^{x_1} + e^{x_2} + e^{x_3} + e^{x_4} + e^{x_5} + e^{x_6}$$

$$\bigg\\{\frac{e^{x_1}}{\text{sum}},\frac{e^{x_2}}{\text{sum}},\frac{e^{x_3}}{\text{sum}},\frac{e^{x_4}}{\text{sum}},\frac{e^{x_5}}{\text{sum}},\frac{e^{x_6}}{\text{sum}},\bigg\\}$$

#### Example: Safe Softmax
$$\\{x_1, x_2, x_3, x_4, x_5, x_6\\}$$

$$\text{sum} = e^{x_1-\text{max}} + e^{x_2-\text{max}} + e^{x_3-\text{max}} + e^{x_4-\text{max}} + e^{x_5-\text{max}} + e^{x_6-\text{max}}$$

$$\bigg\\{\frac{e^{x_1-\text{max}}}{\text{sum}},\frac{e^{x_2-\text{max}}}{\text{sum}},\frac{e^{x_3-\text{max}}}{\text{sum}},\frac{e^{x_4-\text{max}}}{\text{sum}},\frac{e^{x_5-\text{max}}}{\text{sum}},\frac{e^{x_6-\text{max}}}{\text{sum}},\bigg\\}$$



* __Peaky Output__: Softmax gives more attention to higher values and less attention to lower values.
* Probabilities of a row b/w 0-1
* Sum of row values sum = 1

#### Scaling
* $Q \times K^T$ variance can become vary large
* Variance scales as $\sqrt{d_{keys}}$
* We want variance to stay as close to one
* Why divide by $\sqrt{d_{keys}}$

***

1. Normal softmax
2. Safe softmx

*** 

