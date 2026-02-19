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

* 30:00

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

* 45:00

this context. continuous. view what this will do is that this will just
45:44
merge um the first row second row of token one first row second row token two
45:49
first row second row of token number three and it will give me an output context Matrix of size 1A 3 comma 6
45:57
remember now the beauty of this context Vector Matrix is that whenever someone looks at this size they'll just see 1A
46:04
3A 6 but the way we have reached this is that we have OB we actually obtained two context vectors right we obtained two
46:11
context Vector matrices and then we merged them together into one so this
46:16
one final context Vector Matrix actually contains two perspectives it contains perspectives from the head one as well
46:23
as head two so it's much richer than having just the self ention mechanism producing a context Vector Matrix
46:30
because now we actually had multiple context Vector matrices and we merg them together if there were six attention
46:36
heads we would have six context Vector matrices which would be merged together to give me my final context Vector
46:42
Matrix that's the beauty of multi-ad attention although the dimension looks the same as it would have when we did
46:48
self attention but now the Matrix is much more richer since it captures multiple perspectives that's it this is
46:56
the last step of the multi-ad attention and I want to thank you all for sticking through this entire lecture and seeing
47:04
all the steps especially when we look at matrices and dimensions things can get a bit complicated and when you look at
47:10
this code directly you'll you'll think it's a bit complicated right but it's actually very simple if you understand
47:17
the mathematics with respect to matrices then the code actually makes a lot of sense this is the main class the
47:25
multi-ad attention which powers all the major large language models out there of course there were a lot of improvements
47:31
after this such as KV caching multi-head latent attention flash attention Etc but
47:37
if you understand these three dimensions B comma number of tokens comma D in if you understand what the keys. view
47:43
values. view queries. view does what transpose means transpose one comma 2
47:48
how it relates to the handwritten exercise you will you'll find these things are not very difficult so I
47:55
highly encourage all of you to take a piece of paper and write all these things down as as if you're following
48:01
this lecture seriously so once this class is actually defined you can what you can simply do
48:07
is that you can just take an inputs Vector which is 1 2 3 so I have two I
48:13
have three rows over here uh my three tokens and each token
48:19
is a six dimensional Vector the same example which we saw on the notes and then we can just have so here I'm having
48:26
two batches so I have stacked the inputs on top of the inputs so remember
48:31
although in the handwritten notes we just took one batch the code is powerful enough to change the First Dimension to
48:37
even two so that's what I'm considering over here I'm stacking these two inputs one on top of each other to create a
48:43
batch and then I just pass this entire input to this multi-ad attention that's
48:48
it and then I create the context vectors so you'll see the first context Vector
48:53
is 1x 3 comma 6 this is exactly the context Vector shape which we had seen over here um 1x3 comma 6 and the second
49:04
context Vector is 1x 3A 6 so this is the first batch this whole thing is the first batch and this whole thing is the
49:10
second batch that's why we have the size here 2x 3 comma 6 if we had three batches it would have been 3 comma 3
49:16
comma 6 so just within five to six lines of code we have implemented the multi-ad
49:22
attention uh calculation and here if you scroll above these are are 20 to 25
49:27
lines of code which is the mechanism which Powers uh or which is the brain behind
49:35
how why large language models work so well these 25 lines of code actually encode the key advancement which
49:42
happened in 2017 when the Transformer block was introduced for the first time and if
49:49
someone just takes a looks look at this code they'll find it difficult but my main purpose of today's class was to
49:55
link it to um handwritten notes of mathematical derivation and also to an
50:00
intuition which we looked at in the previous class only then I showed you the code so that you don't get scared or
50:06
intimidated by the code but if you're are seriously interested about developing your understanding and never
50:12
forgetting how multi-ad attention Works take a piece of paper write everything down so that you don't forget this at
50:18
all now that we have completed this lecture we have done multi-head attention so we have finished these
50:24
three part we are now fully ready to start learning about key value cache key
50:29
value cache is that main mechanism which made multihead attention much more efficient and this serves as the bridge
50:36
towards finally understanding multi-head latent attention that's the real key Innovation which was implemented in the
50:42
Deep seek paper but to understand key value cach and to understand multi-head latent attention it would have been very
50:48
difficult for you to understand this if you if you did not understand today's lecture so that's why we had all these
50:54
lectures on self attention causal attention multi-head attention so I want to congratulate you and thank you for
51:00
reaching this part please stay with me the later parts will be even more rewarding now that you have finished uh
51:08
completing the lectures until here so thanks a lot everyone please make notes along with me so that you learn the most
51:14
thanks everyone I look forward to seeing you in the next lecture























