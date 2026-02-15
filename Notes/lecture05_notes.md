#### [invideo - Create Videos Without Limits](https://invideo.io/?utm_source=google&utm_medium=cpc&utm_campaign=Top16_Search_Brand_Exact_EN&adset_name=Invideo_AI&keyword=invideo.%20io&network=g&device=c&utm_term=invideo.%20io&utm_content=Invideo_AI&matchtype=e&placement=g&campaign_id=18035330768&adset_id=152533182854&ad_id=674819400171&gad_source=1&gad_campaignid=18035330768&gbraid=0AAAAACqfi_AodnB9BAf3Yt3ThUh5Qgi6U&gclid=EAIaIQobChMIqujNluXbkgMV3UpHAR2hexKwEAAYASAAEgKn4fD_BwE)
* __Prompt__: Create a hyper realistic video commertical of a premium luxury watch, make it cinematic, use closeup of the watch and its parts. Use American female voice for english narration.

***

#### Scaled Dot-Product Attention

$$\text{Attention}(Q,K,V)=\text{softmax}\bigg(\frac{QK^T}{\sqrt{d_{k}}}\bigg)V$$

***

#### Self Attention with Trainable Weights

```
The next day is bright
```

* Input Embedding = Token Embedding + Positional Embedding

####  Example
* Words = 5
* d_input = Input Emdebbing dimenesion = 8
* (5 X 8)

#### Trainable Weight Metrices
* Query  = W_q = (8,4) = (d_input, d_output)
* Keys   = W_k = (8,4)
* Values = W_v = (8,4)

* For W_q, W_k, and W_v usually d_input = d_output

* We want to tranform input embeddings into different space, so that, our expressivity increses and we can capture undering complexitites which cann't be done through a simple dot product. 

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

