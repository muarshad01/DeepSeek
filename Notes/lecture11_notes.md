## Group Query Attention (GQA)
* Multi-Head Attention (MHA) - 400 GB
* Multi-Query Attentino (MQA) - 3 GB
* MQA reduces the KV-cache size by a factor of 128 for DeepSeek

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
| MHA                    | $l \times b \times n \times h \times s \times 2 \times 2$ | 4.5 GB | 400 GB |
| MQA ($n_{heads}=1$)    | $l \times b \times 1 \times h \times s \times 2 \times 2$ |  48 MB | 3 GB |
| GQA ($n \rightarrow g$)| $l \times b \times g \times h \times s \times 2 \times 2$ | 384 MB (g=8) | |

***

| Attention Mechanism Type | Number of unique (K,V)-pairs | KV-cache size | Performance (Context Understanding)|
|---|---|---|---|
| MHA | H - Each head has its own K and V                              | Largest  | __BEST__ |
| GQA | 1 - ALL heads share the same K and V                           | Medium   | Medium |
| MQA | G - Heads are divided into G groups, each group shares K and V | __SMALLEST__ | Worst |

***

#### [Introducing Meta Llama 3: The most capable openly available LLM to date](https://ai.meta.com/blog/meta-llama-3/)
* Llama 3 adopts grouped query attention (GQA) across both the 8B and 70B sizes.

***

* 30:00

* __Golden question__:  Can we reduce KV cache size and still obtain good performance?

***



