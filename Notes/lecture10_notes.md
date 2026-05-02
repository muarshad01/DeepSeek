#### How to solve KV-cache memory problem?
1. Multi-Query Attention (MQA)
* What if ALL the attention heads share the same K & V matrics?

***

* 15:00

* How? All attention heads can have the same K,V matrics.

| Attention Mechanism Type | Size of KV Cache | GPT-3 ($175B, l=96, n_h=96$) Memory Needed | DeepSeek ($n_{h}=128$)  Memory Needed |
|---|---|---|---|
| MHA | $l \times b \times n_{heads} \times h \times s \times 2 \times 2$ | 4.5 GB | 400 GB |
| MQA ($n_{heads}=1$) | $l \times b                  \times h \times s \times 2 \times 2$ | 48 MB | 3GB |

#### MHA
* Consider $n_{heads}=4$:
  * $W_{Q_1} \neq W_{Q_2} \neq W_{Q_3} \neq W_{Q_4} \longrightarrow  Q_1 \neq Q_2 \neq Q_3 \neq Q_4$
  * $W_{K_1} \neq W_{K_2} \neq W_{K_3} \neq W_{K_4} \longrightarrow  K_1 \neq K_2 \neq K_3 \neq K_4$
  * $W_{V_1} \neq W_{V_2} \neq W_{V_3} \neq W_{V_4} \longrightarrow  V_1 \neq V_2 \neq V_3 \neq V_4$

  * $A_1(Q_1 \times K_1^T) \neq A_2(Q_2 \times K_2^T) \neq A_3(Q_3 \times K_3^T) \neq A_4(Q_4 \times K_4^T)$ (Attention Matrices)
  * $C_1(A_1 \times V_1) \neq C_2(A_2 \times V_2) \neq C_3(A_3 \times V_3) \neq C_4(A_4 \times V_4)$ (Context Matrices)
  * $P_1 \neq P_2 \neq P_3 \neq P_4$ (Multiple Perspectives)

#### MQA
* Consider $n_{heads}=4 \rightarrow 1$:
  * $W_{Q_1} \neq W_{Q_2} \neq  W_{Q_3} \neq W_{Q_4} \longrightarrow  Q_1 \neq Q_2 \neq Q_3 \neq Q_4$
  * $W_{K_1}=W_{K_2}=W_{K_3}=W_{K_4} \longrightarrow  K_1=K_2=K_3=K_4=K$ (We only need to cache one K)
  * $W_{V_1}=W_{V_2}=W_{V_3}=W_{V_4} \longrightarrow  V_1=V_2=V_3=V_4=V$ (We only need to cache one V)

  * $A_1(Q_1 \times K^T) \neq A_2(Q_2 \times K^T) \neq A_3(Q_3 \times K^T) \neq A_4(Q_4 \times K^T)$ (Attention Matrices)
  * $C_1(A_1 \times V) \neq C_2(A_2 \times V) \neq C_3(A_3 \times V) \neq C_4(A_4 \times V)$ (Context Matrices)
  * $P_1 \neq P_2 \neq P_3 \neq P_4$ (Multiple Perspectives, but reduced accuracy!)

***

#### Disadvantage of MQA
* Significant performance degradation (We'll have reduced accuracy).
* We still capture multiple perspective, however, the ablity to capture multiple perspectives has reduced.
* We are still capturing multiple perspecive, but since the number of parameters have reduced the nuances that we are capturing have reduced.

***

#### Advantage
* KV Cache size is reduced
* Number of trainable parameters are reduced

***

#### DeepSeek has 128 attention heads!
* l : 61
* b : 1
* $n_{heads}$ : 128
* $h : (d_{head})$ : 128
* s : 100,000 (tokens)

***

* 25:00

* [FALCON](https://huggingface.co/docs/transformers/en/model_doc/falcon)

***
