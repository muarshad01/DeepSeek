
| Model | Context Size|
|---|---|
| gpt-4     |  8k |
| gpt-4-32k | 32k |

***

* 30:00

* to generate a new token, we only need the hidden state of the most recent token. None of the other hidden states are required.

***

* 45:00
  
* l.n.h.s.b.2.2
  * l : number of transformer blocks layers
  * $n_{heads}$ : number of attention heads
  * h : attention head dimension ($d_{head}$)
  * s : sequence length (context length)
  * b : batch size
  * 2 : number of byptes per PF (Assume each parameter takes 2 bytes)

  * 2 : Two caches one each for (k,v)

***

#### Size of KV_Cache
$\text{Bytes ~taken ~up ~by ~KV cache!} = l \times b \times s \times h \times n \times 2 \times 2$
