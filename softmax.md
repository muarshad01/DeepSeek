#### $\text{Attention}(Q,K,V)$ 
$$
\begin{align}
   S &= \text{casual mask}\bigg(\frac{Q \times K^T}{\sqrt{d_{keys}}}\bigg)\\
   P &= \text{softmax}\bigg(\text{casual mask}\bigg(\frac{Q \times K^T}{\sqrt{d_{keys}}}\bigg)\bigg)\\
   O &= \text{softmax}\bigg(\text{casual mask}\bigg(\frac{Q \times K^T}{\sqrt{d_{keys}}}\bigg)\bigg) \times V\\
\end{align}
$$

***

## softmax

#### Example: Normal Softmax
$$
\begin{align}
      &= \\{s_1, s_2, s_3, s_4, s_5, s_6\\} \\
      &= \bigg\\{\frac{e^{s_1}}{\text{sum}},\frac{e^{s_2}}{\text{sum}},\frac{e^{s_3}}{\text{sum}},\frac{e^{s_4}}{\text{sum}},\frac{e^{s_5}}{\text{sum}},\frac{e^{s_6}}{\text{sum}},\bigg\\} \\ 
   \text{sum} &= e^{s_1} + e^{s_2} + e^{s_3} + e^{s_4} + e^{s_5} + e^{s_6} \\
\end{align}
$$


#### Example: Safe Softmax
$$
\begin{align}
 &= \\{x_1, x_2, x_3, x_4, x_5, x_6\\} \\
  &= \bigg\\{\frac{e^{x_1-\text{max}}}{\text{sum}},\frac{e^{x_2-\text{max}}}{\text{sum}},\frac{e^{x_3-\text{max}}}{\text{sum}},\frac{e^{x_4-\text{max}}}{\text{sum}},\frac{e^{x_5-\text{max}}}{\text{sum}},\frac{e^{x_6-\text{max}}}{\text{sum}},\bigg\\} \\
m  &= \max\\{x_1, x_2, x_3, x_4, x_5, x_6\\} \\
 \text{sum} &= e^{x_1-\text{m}} + e^{x_2-\text{m}} + e^{x_3-\text{m}} + e^{x_4-\text{m}} + e^{x_5-\text{m}} + e^{x_6-\text{m}} \\
\end{align}
$$


* __Peaky Output__:
   * Softmax gives more attention to higher values and less attention to lower values.
   * Probabilities of a row b/w 0-1
   * Sum of row values sum = 1

#### Variance Issue with increasing dimension
* $Q \times K^T$ variance can become vary large
* Variance scales as $\sqrt{d_{keys}}$
* We want variance to stay as close to one
* Why divide by $\sqrt{d_{keys}}$

***

1. Normal softmax
2. Safe softmx

*** 

