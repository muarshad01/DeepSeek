#### Example
* __Step-1__: Start with 1 input batch
* $X: (1, 3, 6)$
* $(b, s, h) = (b, s, d_{in})$ = (1, 3, 6)
  * batch = 1
  * s = num of tokens = 3
  * h = hidden dim = $d_{in} = 6$

```python
b, s, s = x.shape
```

***

* 10:00
  
* __Step-2__: Decide $(d_{out}, n_{heads}) = (6, 2)$

$$d_{head} = \frac{d_{out}}{n_{head}} = \frac{6}{2} = 4$$

* __Step-3__: Initialize trainable weight matrices for Key, query, value $(W_K, W_Q, W_V)$ 
  * $W_K (d_{in}, d_{out}) = (6, 6)$
  * $W_Q (d_{in}, d_{out}) = (6, 6)$
  * $W_V (d_{in}, d_{out}) = (6, 6)$ 

* __Step-4__: Calculate Q, K, and V matrices: 
  * $Q (b, s, d_{out}) = x \times W_Q = (1, 3, 6)$
  * $K (b, s, d_{out}) = x \times W_K = (1, 3, 6)$
  * $V (b, s, d_{out}) = x \times W_V = (1, 3, 6)$ 

***

* 15:00

* __Step-5__: Unroll last dimension of Keys, Queries, and Values to include num_heads and head_dim
* Unroll last dim: $(b, s, d_{out}) \rightarrow (b, s, n_{heads}, n_{head}) = (1, 3, 2, 3)$

$$\text{head-dim} = \frac{d_{out}}{\text{num-heads}} = \frac{6}{2} = 3$$

***

* 25:00

* __Step-6__: Group matrices by "number of heads"
* (b, num_tokens, num_heads, head_dim) - > (b, num_heads, num_tokens, head_dim)
* (1, 3, 2, 3) -> (1, 2, 3, 3)

***

* 35:00

* __Step-8__: Find attention weights
  * Mask the scores to implement casual attention
  * Dive by $$\sqrt{\text{head-dim}} = \sqrt{\frac{d_{out}}{\text{num-heads}}} = \sqrt{\frac{6}{2}}=\sqrt{3}$$


***

* 40:00

* __Step-9__: Context Vector = Attention Weights X Values
  * (b, num_heads, num_tokens, num_tokens) X (b, num_heads, num_tokens, head_dim)

***
