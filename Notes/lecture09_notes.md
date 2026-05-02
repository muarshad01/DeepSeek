
| Model | Context Size|
|---|---|
| gpt-4     |  8k |
| gpt-4-32k | 32k |

***

* 30:00

* to generate a new token, we only need the hidden state of the most recent token. None of the other hidden states are required.

***

* 45:00
  
#### Size of KV_Cache
* l : number of transformer blocks layers
* b : batch size
* s : sequence length (context length)
* h : attention head dimension ($d_{head}$)
* $n_{heads}$ : number of attention heads
* 2 : number of byptes per PF (Assume each parameter takes 2 bytes)
* 2 : Two caches one each for (k,v)

$$\text{Bytes ~taken ~up ~by ~KV cache} = l \times b \times s \times h \times n_{heads} \times 2 \times 2$$

***

* [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/): Can we store KV-cache in low precisin. 

***

#### MHA
* Consider $n_{heads}=4$:
  * $W_{Q_1} \neq W_{Q_2} \neq W_{Q_3} \neq W_{Q_4} \longrightarrow  Q_1 \neq Q_2 \neq Q_3 \neq Q_4$
  * $W_{K_1} \neq W_{K_2} \neq W_{K_3} \neq W_{K_4} \longrightarrow  K_1 \neq K_2 \neq K_3 \neq K_4$
  * $W_{V_1} \neq W_{V_2} \neq W_{V_3} \neq W_{V_4} \longrightarrow  V_1 \neq V_2 \neq V_3 \neq V_4$

  * $A_1(Q_1 \times K_1^T) \neq A_2(Q_2 \times K_2^T) \neq A_3(Q_3 \times K_3^T) \neq A_4(Q_4 \times K_4^T)$ (Attention Matrices)
  * $C_1(A_1 \times V_1) \neq C_2(A_2 \times V_2) \neq C_3(A_3 \times V_3) \neq C_4(A_4 \times V_4)$ (Context Matrices)
  * $P_1 \neq P_2 \neq P_3 \neq P_4$ (Multiple Perspectives)


***
