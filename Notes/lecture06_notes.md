#### Causal attention
* Causal attention, also known as a masked attention is a special form of self attention.

***

* 10:00

1. Causal attention, also known as a masked attention is a special form of self attention.
2. It restricts the model to only consider previous and current inputs in a sequence, when processing any given token.
3. This in contrast to self attention mechanism, which allows access to the entire input sequence at once.
4. When computing attention scores, the casual attention mechanism ensures that the model
5. To achieve this in GPT like LLMs, for each token processed, we mask out the futhre tokens, which come after the current token in the input text.

* Also called masked attention; unidirectional attention (autoregressive).

***

* 20:00

* We mask out the attention weights above the diagonal, and we normalize the non-masked attention weights, such that the attention weights sum upto 1 in each row.

***

* 25:00

* Attention score -> softmax -> attention weights -> Add 0's above diagonal -> masked attention score -> normalize rows -> masked attention weights
  
***

* 35:00

* Masking additional attention weights with dropout
* Dropout is a deep learning technique where randomly selected hidden layer units are ignored during training.
* This prevents overfitting and improves generalization performance.

***
