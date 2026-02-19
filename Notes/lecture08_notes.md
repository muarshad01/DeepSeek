#### Example
* __Step-1__: Start with 1 input batch
* X: (1, 3, 6)
* (b, num_tokens, d_in) = (1, 3, 6)
  * batch = 1
  * num_tokens = 3
  * d_in = 6

```python
b, num_tokens, d_in = x.shape
```

***

* 10:00
  
* __Step-2__: Decide (d_out, num_heads) = (6, 2)

$$\text{head-dim} = \frac{d_{out}}{\text{num-heads}} = \frac{6}{2} = 3$$

* __Step-3__: Initialize trainable weight matrices for Key, query, value (W_k, W_q, W_v) 
  * W_k (d_in, d_out) = (6, 6)
  * W_q (d_in, d_out) = (6, 6) 
  * W_v (d_in, d_out) = (6, 6) 

* __Step-4__: Calculate Keys, Queries, Value Matrix (Input X W_k, Input X W_q, Input X W_v)
  * Keyes (b, num_tokens, d_out) = (1 X 3 X 6) 
  * Queries (b, num_tokens, d_out) = (1 X 3 X 6) 
  * Values (b, num_tokens, d_out) = (1 X 3 X 6) 

***

* 15:00

* __Step-5__: Unroll last dimension of Keys, Queries, and Values to include num_heads and head_dim
* Unroll last dim: (b, num_tokesn, d_out) -> (b, num_tokesn, num_heads, head_dim) = (1, 3, 2, 3)

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
