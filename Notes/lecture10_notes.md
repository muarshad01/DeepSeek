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
  * $ W_{k1}=W_{k2}=W_{k3}=W_{k4}$ (Weight Matrices)
  * $A_1=A_2=A_3=A_4$ (Attention Matrices)
  * $V_1=V_2=V_3=V_4$ (Value Matrices)
  * $C_1=C_2=C_3=C_4$ (Context Matrices)
  * $P_1=P_2=P_3=P_4$ (Perspectives)

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










