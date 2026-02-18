#### problem with the self attention mechanism is and
* The artist painted the portrait of a woman with a brush.
1. painting a woman with a "brush"
2. painting of a "woman with a brush"

***

* 10:00

* Self attention can only capture a single perspective in a given input sequence. It cann't capture multiple perspectives.

***

* 20:00

| Self Attention | Perspective |
|---|---|
| 1 Self Attention | 1 Perspective | 1 Context Vector Matrix | 
| Multiple Self Attention | Multiple Perspective | Multiple Context Vector Matrices |

***

* 25:00

#### Implementing a 2-head attention (step-by-step)

1. Input Embedding
* Example: (11 X 8), where d_in=8
2. Start with a single W_q, W_k, W_v
* Example: (8 X 4), where d_out=t
* Output is (11x4) for query vectors, key vectors, value vectors
3. Split W_q, W_k, W_v into multiple heads
* Example: (8 X 2) (W_q1,W_q2), (W_k1,W_k2), (W_v1,W_v2)

$$\text{head-dim} = \frac{d_{out}}{num ~of ~heads} = \frac{4}{2} = 2$$

***

* 35:00

6. Computing attentin weights for each head
* Scaling - softmax - causal attantion - Dropout
7. Merge the contex matrix for two heads

***
