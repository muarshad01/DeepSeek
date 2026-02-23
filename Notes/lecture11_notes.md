## Group Query Attention (GQA)

* Normal Multi-Head Attention (MHA) - 400 GB
* Multi Query Attentino (MQA) - 3 GB
* MQA reduces the KV size by a factor of 128 for DeepSeek

#### Disadvantages of MQA
* Significant performance degradation
* Remember the purpose of MHA: each head captures a different perspective!
* MQA: Diversity across multiple heads is very low!!!

***

* 10:00

1. MHA (ALL heads are different) - Complete diversity!
2. GQA
3. MQA (ALL heads are same) - Least diversity - performance degradation!

* GQA: Instead of all attention heads kaving same K-V matrices, what if we create groups of attention heads.

***

* 15:00

#### Diversity
* MHA >> GQA >> MQA

#### Memory
* MHA << GQA << MQA

***

* 20:00


| Attention Mechanism Type | Size of KV Cache | GPT-3 (175B, l=96, n=96) memory needed | DeepSeek (n=128)  memory needed|
|---|---|---|---|
| Multi-Head Attention (MHA)  | l.b.n.h.s.2.2           | 4.5 GB            | 400 GB |
| Multi-Query Attention (MQA) | l.b.1.h.s.2.2 (n=1)     |  48 MB            |   3 GB |
| Group-Query Attention (GQA) | l.b.g.s.2.2  (n -> g)   | 384 MB (8 groups) |        |

***

| Attention Mechanism Type | Number of unique key-value (KV) pairs | KV cache size | Performance (Context Understanding)|
|---|---|---|---|
| Multi-Head Attention (MHA)    | H (Each head has its own K and V)                              | Largest  | __BEST__ |
| Grouped-Query Attention (GQA) | 1 (ALL heads share the same K and V)                           | Medium   | Medium |
| Multi-Query Attention (MQA)   | G (Heads are divided into G groups, each group shared K and V) | __SMALLEST__ | Worst |


#### [Introducing Meta Llama 3: The most capable openly available LLM to date](https://ai.meta.com/blog/meta-llama-3/)
* Llama 3 adopts grouped query attention (GQA) across both the 8B and 70B sizes.

***

* 25:00

*** 

* 30:00

* __Golden question__:  Can we create an __Attention Mechanism Type__ for which not only the KV cache size is  low but also the performance is very good.



Can I create something like this? Essentially I want
34:10
performance to be as good as the multi head attention and I want my KV cache to
34:16
also be very small. So can I have the best of both worlds and it looks like an impossible problem to solve right? If
34:23
you try to have a good performance, you will of course have to have more memory
34:28
and but they solve this problem by introducing a very beautiful trick which is the multi head latent attention. In
34:35
the next lecture, we are finally going to start looking at multi head latent attention in a lot of detail and uh
34:42
there is also a coding module which I have in which we'll try to implement the multi head latent attention from
34:47
scratch. I know we have been building up to this lecture for quite some time now but finally we are at this stage where
34:54
we have literally covered everything. We covered self attention, we covered causal attention, we covered multi head
35:00
attention, we covered key value cache, we covered MQA, we covered GQA. The only
35:05
thing which is now remaining is to finally tackle the key innovation in the deepseek architecture. That's the multi
35:11
head latent attention. This is what we'll be seeing in the next lecture. As the lecture gets more lectures get more
35:17
deep and technical, I encourage you to maintain notes, maintain detailed notes of what you are learning so that you
35:24
don't feel lost along the way. And until now it was very important to explain all these concepts to you because otherwise
35:31
it will be very hard to understand latent attention and the beauty of what deepse did they really got the best of
35:38
both worlds. They reduced the memory requirements of the KV cache and they also got a great
35:45
performance out of this mechanism called multi head latent attention which we'll look at in the next lecture. So thanks
35:52
everyone and I look forward to seeing you in the next lecture.

















