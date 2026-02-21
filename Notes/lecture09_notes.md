
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
* l : number of transformer blocks
* n : number of attention heads
* h : attention head size
* s : context length
* b : batch size
* 2 : Two caches one each for (k,v)
* 2 : number of byptes per floating point

***
