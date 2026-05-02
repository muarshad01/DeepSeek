#### Group Query Attention (GQA)

* Consider $n_{heads}=4 \rightarrow 1$:
  * $W_{Q_1} \neq W_{Q_2} \neq  W_{Q_3} \neq W_{Q_4} \longrightarrow  Q_1 \neq Q_2 \neq Q_3 \neq Q_4$
  * $(W_{K_1}=W_{K_2})  ~and (W_{K_3}=W_{K_4}) \longrightarrow  (K_1=K_2)   ~and~  (K_3=K_4=K)$ 
  * $(W_{V_1}=W_{V_2})  ~and (W_{V_3}=W_{V_4}) \longrightarrow  (V_1=V_2)   ~and~  (V_3=V_4=V)$ 

  * $A_1(Q_1 \times K^T) \neq A_2(Q_2 \times K^T) \neq A_3(Q_3 \times K^T) \neq A_4(Q_4 \times K^T)$ (Attention Matrices)
  * $C_1(A_1 \times V) \neq C_2(A_2 \times V) \neq C_3(A_3 \times V) \neq C_4(A_4 \times V)$ (Context Matrices)
  * $P_1 \neq P_2 \neq P_3 \neq P_4$ (Multiple Perspectives, but reduced accuracy!)

***


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



