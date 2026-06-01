#### Multi-Token Prediction (MTP) Coded from Scratch
* **Step-0**: Load Packages
* **Step-1**: Define class RMSNorm
* **Step-2**: Define the Multi-token Prediction (MTP) class
* **Step-3**: Pass input tokens through the model and generate multiple next tokens

$$(B, S, 3, V) = (B, T-D, 3, V)$$

***

* 30:00

* __Step-4__: Calculate loss between target tokens and predicted tokens

#### Cross Entropy Loss
* L=5
* D=3

$$
\begin{align}
i&=1 \rightarrow [PT1 \leftrightarrow TT1, PT2 \leftrightarrow TT2,PT3 \leftrightarrow TT3] \rightarrow L_1\\
i&=2 \rightarrow [PT2 \leftrightarrow TT2,PT3 \leftrightarrow TT3,PT4 \leftrightarrow TT4] \rightarrow L_2\\
i&=3 \rightarrow [PT3 \leftrightarrow TT3, PT4 \leftrightarrow TT4,PT5 \leftrightarrow TT5] \rightarrow L_3\\
i&=4 \rightarrow [PT4 \leftrightarrow TT4, PT5 \leftrightarrow TT5,PT6 \leftrightarrow TT6] \rightarrow L_4\\
i&=5 \rightarrow [PT5 \leftrightarrow TT5, PT6 \leftrightarrow TT6,PT7 \leftrightarrow TT7] \rightarrow L_5\\
\end{align}
$$

$$
Loss = \frac{1}{15}(L_1+L_2+L_3+L_4+L_5)
$$

***
