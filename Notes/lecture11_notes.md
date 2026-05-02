#### Group Query Attention (GQA)
* GQA: Instead of ALL attention heads kaving same KV-Cache matrices (MQA), what if we create groups of attention heads.
* I still want my model to retain multiple perspectives, but I also want to to reduce KV-Cache size!

***

* Consider $n_{heads}=4$ divided in two groups $G_1$ and $G_2$:
  * $W_{Q_1} \neq W_{Q_2} \neq W_{Q_3} \neq W_{Q_4}\longrightarrow Q_1 \neq Q_2 \neq Q_3 \neq Q_4$
  * $G_1 (Q_1, Q_2); G_2 (Q_3, Q_4)$
  * $(W_{K_1}=W_{K_2}) \longrightarrow (K_1=K_2=K_A)$
  * $(W_{K_3}=W_{K_4}) \longrightarrow (K_3=K_4=K_B)$ 
  * $(W_{V_1}=W_{V_2}) \longrightarrow (V_1=V_2=V_A)$
  * $(W_{V_3}=W_{V_4}) \longrightarrow (V_3=V_4=V_B)$ 

  * $A_1(Q_1 \times K_A^T) \neq A_2(Q_2 \times K_A^T)$
  * $A_3(Q_3 \times K_B^T) \neq A_4(Q_4 \times K_B^T)$
  * $C_1(A_1 \times V_A) \neq C_2(A_2 \times V_B)$
  * $C_3(A_3 \times V_A) \neq C_4(A_4 \times V_B)$ 
  * $P_1 \neq P_2$
  * $P_3 \neq P_4$
  * Each group has a different perspective!

* NOTE: We're not reducing the number of heads. We're justh sharing the parameters across heads.

***

* 20:00


|Reduction Factor| Attention Mechanism Type | Size of KV Cache | GPT-3 $(175B, l=96, n_{heads}=96)$ Memory Needed | DeepSeek $(n_{heads}=128)$ Memory Needed|
|---|---|---|---|---|
|| MHA                    | $l \times b \times n_{heads} \times h \times s \times 2 \times 2$ | 4.5 GB | 400 GB |
|$\frac{1}{n}$| MQA ($n_{heads}=1$)    | $l \times b \times 1 \times h \times s \times 2 \times 2$ |  48 MB | 3 GB |
|$\frac{g}{n}$| GQA ($n \rightarrow g$)| $l \times b \times g \times h \times s \times 2 \times 2$ | 384 MB (g=8) | |


#### Diversity
* MHA >> GQA >> MQA

#### Memory
* MHA << GQA << MQA

***


***

|Reduction Factor| Attention Mechanism Type | Number of unique (K,V)-pairs | KV-cache size | Performance (Context Understanding)|
|---|---|---|---|---|
|| MHA | H - Each head has its own K and V                              | Largest  | __BEST__ |
| $\frac{g}{n}$ | GQA | 1 - ALL heads share the same K and V                           | Medium   | Medium |
| $\frac{1}{n}$ | MQA | G - Heads are divided into G groups, each group shares K and V | __SMALLEST__ | Worst |

***

#### [Introducing Meta Llama 3: The most capable openly available LLM to date](https://ai.meta.com/blog/meta-llama-3/)
* Llama 3 adopts grouped query attention (GQA) across both the 8B and 70B sizes.

***

* 30:00

* __Golden question__:  Can we reduce KV cache size and still obtain good performance?

***



