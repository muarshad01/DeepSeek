
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
