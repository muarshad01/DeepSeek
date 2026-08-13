## Scaled Dot-Product Attention

#### [invideo - Create Videos Without Limits](https://invideo.io/?utm_source=google&utm_medium=cpc&utm_campaign=Top16_Search_Brand_Exact_EN&adset_name=Invideo_AI&keyword=invideo.%20io&network=g&device=c&utm_term=invideo.%20io&utm_content=Invideo_AI&matchtype=e&placement=g&campaign_id=18035330768&adset_id=152533182854&ad_id=674819400171&gad_source=1&gad_campaignid=18035330768&gbraid=0AAAAACqfi_AodnB9BAf3Yt3ThUh5Qgi6U&gclid=EAIaIQobChMIqujNluXbkgMV3UpHAR2hexKwEAAYASAAEgKn4fD_BwE)
* __Prompt__: Create a hyper realistic video commertical of a premium luxury watch, make it cinematic, use closeup of the watch and its parts. Use American female voice for english narration.

***

$$\text{Attention}(Q,K,V)=\text{softmax}\bigg(\frac{QK^T}{\sqrt{d_{k}}}\bigg)V$$

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

#### Query, Key, and Value Vectors
* $Q(5,4) = x \times W_q$
* $K(5,4) = x \times W_k$
* $V(5,4) = x \times W_v$



* We want to tranform input embeddings X(5,8) into different space X(5,4), so that, our expressivity increses and we can capture underline complexities, which can't be done through a simple dot product. 

***

* 20:00

* $$\text{Attention ~Score} = Q \times K^{T}$$

***

* 25:00

#### Softmax

$$\\{x_1, x_2, x_3, x_4, x_5, x_6\\}$$

$$\bigg\\{\frac{e^{x_1}}{\text{sum}},\frac{e^{x_2}}{\text{sum}},\frac{e^{x_3}}{\text{sum}},\frac{e^{x_4}}{\text{sum}},\frac{e^{x_5}}{\text{sum}},\frac{e^{x_6}}{\text{sum}},\bigg\\}$$

$$\text{sum} = e^{x_1} + e^{x_2} + e^{x_3} + e^{x_4} + e^{x_5} + e^{x_6}$$


* __Peaky Output__: Softmax gives more attention to higher values and less attention to lower values.
* __Unstable Training__:

#### Scaling
* Why $\sqrt{d_{keys}}$
* Variance scales as $\sqrt{d_{keys}}$

***

* 35:00

* Attention Score
* Attentin Weight (are normalized)

***


