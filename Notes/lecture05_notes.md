## Scaled Dot-Product Attention

#### [invideo - Create Videos Without Limits](https://invideo.io/?utm_source=google&utm_medium=cpc&utm_campaign=Top16_Search_Brand_Exact_EN&adset_name=Invideo_AI&keyword=invideo.%20io&network=g&device=c&utm_term=invideo.%20io&utm_content=Invideo_AI&matchtype=e&placement=g&campaign_id=18035330768&adset_id=152533182854&ad_id=674819400171&gad_source=1&gad_campaignid=18035330768&gbraid=0AAAAACqfi_AodnB9BAf3Yt3ThUh5Qgi6U&gclid=EAIaIQobChMIqujNluXbkgMV3UpHAR2hexKwEAAYASAAEgKn4fD_BwE)
* __Prompt__: Create a hyper realistic video commertical of a premium luxury watch, make it cinematic, use closeup of the watch and its parts. Use American female voice for english narration.

***

#### Self Attention with Trainable Weights

```
The next day is bright
```


####  Example
* X (5,8)
* Number of words = 5
* Embedding dimension = 8
* Input Embedding = Token Embedding + Positional Embedding

```
       | 1 2 3 4 5 6 7 8 |
The    |                 |
next   |                 |
day    |                 |
is     |                 |
bright |                 |
```

#### Query, Key, and Value Weight Matrices
* $W_q = (8,4)$ 
* $W_k = (8,4)$
* $W_v = (8,4)$

#### Query, Key, and Value Matrices
* $Q(5,4) = x \times W_q$
* $K(5,4) = x \times W_k$
* $V(5,4) = x \times W_v$

* We want to tranform input embeddings X(5,8) into different space X(5,4), so that, our expressivity increses and we can capture underline complexities, which can't be done through a simple dot product. 

***

* 1:20:00

$$
\begin{align}
       \text{Attention ~Score}(5,5)         &= Q \times K^{T} \\
       \text{Attention Weight}(5,5)         &=\text{softmax}\bigg(\frac{QK^T}{\sqrt{d_{k}}}\bigg) \\
       \text{Context Vector Matrix}(5,4)    &=\text{softmax}\bigg(\frac{QK^T}{\sqrt{d_{k}}}\bigg)V \\
\end{align}
$$

***

$$\text{Attention}(Q,K,V)=\text{softmax}\bigg(\frac{QK^T}{\sqrt{d_{k}}}\bigg)V$$

#### Softmax

***

* 35:00

* Attention Score
* Attentin Weight (are normalized)

***


