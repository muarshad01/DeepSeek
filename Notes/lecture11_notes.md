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
| Grouped-Query Attention (GQA) | l.b.g.s.2.2  (n -> g)   | 384 MB (8 groups) |        |

***

| Attention Mechanism Type | Number of unique key-value (KV) pairs | KV cache size | Performance (Context Understanding)|
|---|---|---|---|
| Multi-Head Attention (MHA)    | H (Each head has its own K and V)                              | Largest  | __BEST__ |
| Grouped-Query Attention (GQA) | 1 (ALL heads share the same K and V)                           | Medium   | Medium |
| Multi-Query Attention (MQA)   | G (Heads are divided into G groups, each group shared K and V) | __SMALLEST__ | Worst |


#### [Introducing Meta Llama 3: The most capable openly available LLM to date](https://ai.meta.com/blog/meta-llama-3/)
* Llama 3 adopts grouped query attention (GQA) across both the 8B and 70B sizes.

***

* 30:00

* __Golden question__:  Can we reduce KV cache size and still obtain good performance?

***



