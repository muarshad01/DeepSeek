#### Sinusoidal Positional Encoding (SinPE)

$$
\begin{aligned}
    PE_{(pos, 2i)}   &= sin\bigg(\frac{pos}{100000^{\frac{2i}{d_{model}}}}\bigg)\\
    PE_{(pos, 2i+1)} &= cos\bigg(\frac{pos}{100000^{\frac{2i}{d_{model}}}}\bigg)\\
\end{aligned}
$$

* (pos, index)
  * Event index = $$2i$$
  * Odd index = $$2i+1$$
  * The values are in the range [-1, 1]

* [Attention Is All You Need - 2017](https://arxiv.org/abs/1706.03762)

***

* 15:00

#### EXAMPLE GPT-2
* Context size = 1024
    * The range of "pos" variable is [1, 1024]
* Embedding dimentions = $$d_{model}$$ = 768
    * The range of "i" or index variable is [1, 768]

***

* 20:00

* SinPE avoid the discontinuous nature that was there with BPE.
* SinPE values are smooth, continuous, and also differentiable.
* Thats helps to a much stable LLM optimization routine.

***

* 25:00

* __Property 2__: Linear relation between two encoded positions.

***

* 30:00

***

* 35:00

* Relative positional encodings are just rotations of each other.

* rotations ensure that relative ship shifts map to fixed angular differences, which then translate into predictable, learnable attention patterns - like focusing more on nearby words.

#### What's the main problem with sinusoidal embedding?
* One major issue is that we add these encodings directly to token embeddings. This can __pollute the semantic information__ carried by token embeddings.


* can we instead augment my query and the key vectors itself with positional embeddings?

***

* 40:00

#### Rotary Positional Encoding (RoPE)

* The main idea is to take Query and Key vectors, and to apply the sine and cosine  positional encoding to these vectors.

***

