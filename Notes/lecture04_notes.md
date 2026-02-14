#### The need for attention mechanism

* Here is how the field of Generative AI has evolved

| Model | Year | (Encoder, Decoder) |
|---|---|---|
| RNN | 1980 |
| LSTM | 1997 |
| Attention + RNN | 2014 |
| Attention + Transformer | 2017 | (Encoder, Decoder)| 
| BERT | 2018 | (Encoder, ---)|
| Attention + GPT | 2018 | (--, Decoder)

***

* [ChatGPT](https://chatgpt.com/)
  * I want to learn about AI. Can you help me?

***

* 10:00

* NN can't deal with memory.
* Context

***

* 15:00

* Context bottleneck in RNN is not good for retaining long-range context.

***

* 20:00

* we need to selectively access parts of the input sequence during decoding.
* __Context Window__

***

* 25:00

* while decoding, we can quantify how much  importance (attention) needs to be given to each input token.

* [Neural Machine Translation by Jointly Learning to Align and Translate - 2014](https://arxiv.org/abs/1409.0473)
  * The is "Bahdanau" Attention Mechanism.
  * RNN + Attention Mechanism
* [Attention Is All You Need - 2017](https://arxiv.org/abs/1706.03762)
  * RNN architecture is NOT required for building DNN (they have context problem)
  * Transformer Architecture


***

* 30:00

#### Self Attention
* Mechanism which allows each position in the input sequence to attend to all positions in same sequence.

***

* 35:00

***

* 40:00

* __Context Vector__

* Context vector is an enriched embedding vector. It combines informatin from all other input elements.

***

* 45:00

* $$\\{x^{1}, x^{2}, x^{3}, x^{4}, x^{5}\\}$$

* $$z^{2}= \alpha_{21} \times x^{1} + \alpha_{22} \times x^{2} + \alpha_{23} \times x^{3} +
\alpha_{24} \times x^{4} + \alpha_{25} \times x^{5}$$
* $$z^{2}$$ [context vecot for $$x^{2}$$]

* What's wrong with this approach?

```
The dog chased the ball, but it couldn't catch it.
```

* Simple dot product only measures basic semantic similarity, which isn't sufficient to resolve more nuanced contextual ambiguities.

***

* 50:00

#### Option-2
* Insted of directly using embeddings, we transform each embedding through trained matrices.

*** 
