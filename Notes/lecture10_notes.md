#### How to solve KC cache memory problem?
1. Multi-Query Attention (MQA)
* What if all the attention heads share same Key & Value matrics?
3. Group-Query Attention (GQA)

***

* 15:00

* How? All attention heads can have the same K,V matrics.

| Attention Mechanism Type | Size of KV Cache | GPT-3 ($175B, l=96, n_h=96$) Memory Needed | DeepSeek ($n_{h}=128$)  memory needed||
|---|---|---|---|---|
| MHA | $l \times b \times n_{heads} \times h \times s \times 2 \times 2$ | 4.5 GB | 400 GB | Multiple Perspectives |
| MQA ($n_{heads}=1$) | $l \times b                  \times h \times s \times 2 \times 2$ | 48 MB | 3GB | Single Perspective |

#### Example
* If, for MHA ($n_{heads}=4$) and for MQA($n_{heads}=1$) then:
  * $W_{Q_1} \neq W_{Q_2} \neq  W_{Q_3} \neq W_{Q_4} \longrightarrow  Q_1 \neq Q_2 \neq Q_3 \neq Q_4$
  * $W_{K_1}=W_{K_2}=W_{K_3}=W_{K_4} \longrightarrow  K_1=K_2=K_3=K_4=K$
  * $W_{V_1}=W_{V_2}=W_{V_3}=W_{V_4} \longrightarrow  V_1=V_2=V_3=V_4=V$

  * $A_1(Q_1 \times K^T) \neq A_2(Q_2 \times K^T) \neq A_3(Q_3 \times K^T) \neq A_4(Q_4 \times K^T)$ (Attention Matrices)
  * $C_1(A_1 \times V) \neq C_2(A_2 \times V) \neq C_3(A_3 \times V) \neq C_4(A_4 \times V)$ (Context Matrices)
  * $P_1 \neq P_2 \neq P_3 \neq P_4$ (Perspectives)
* We still capture multiple perspective, however, the ablity to capture multiple perspectives has reduced.
* We are still capturing multiple perspecive, but since the number of parameters have reduced the nuances that we are capturing have reduced.
* We'll reduce accuracy.

***

#### DeepSeek has 128 attention heads!
* L : 61
* b : 1
* n : 128
* h : 128
* s : context-length : 100,000 (Number of tokens?)

* __Each query still has its own projection (like in MHA).__
* __All queries share the same key and value vectors.__

***

* 20:00

#### Disadvantage of MQA
* Significant performance degradation
* Remember the purpose of MHA: Each head captures different perspectives!

***

* 25:00

* [FALCON](https://huggingface.co/docs/transformers/en/model_doc/falcon)

***
