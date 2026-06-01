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
$$
i=1 \rightarrow [PT1 \leftrightarrow TT1, PT2 \leftrightarrow TT2,PT3 \leftrightarrow TT3] \rightarrow L_1\\
i=2 \rightarrow [PT1 \leftrightarrow TT1, PT2 \leftrightarrow TT2,PT3 \leftrightarrow TT3] \rightarrow L_2\\
i=3 \rightarrow [PT1 \leftrightarrow TT1, PT2 \leftrightarrow TT2,PT3 \leftrightarrow TT3] \rightarrow L_3\\
i=4 \rightarrow [PT1 \leftrightarrow TT1, PT2 \leftrightarrow TT2,PT3 \leftrightarrow TT3] \rightarrow L_4\\
i=5 \rightarrow [PT1 \leftrightarrow TT1, PT2 \leftrightarrow TT2,PT3 \leftrightarrow TT3] \rightarrow L_5\\
$$

***
