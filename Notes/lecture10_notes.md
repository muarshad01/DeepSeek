#### How to solve KC cache memory problem?
1. Multi-Query Attention (MQA)
* What if all the attention heads share same Key & Value matrics?
3. Group-Query Attention (GQA)

***

* 15:00

* How? All attention heads can have the same K,V matrics.

| Attention Mechanism Type | Size of KV Cache | GPT-3 (175B, l=96, n=96) memory needed | DeepSeek (n=128)  memory needed|
|---|---|---|---|
| Multi-Head Attention (MTA)  | l.b.n.h.s.2.2 | 4.5 GB | 400GB |
| Group-Query Attention (GQA) | l.b.h.s.2.2   | 48 MB | 3GB |
 
#### DeepSeek has 128 attention heads!
* L : 61
* b : 1
* n : 128
* h : 128
* s : 100,000 (Number of tokens?)

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






