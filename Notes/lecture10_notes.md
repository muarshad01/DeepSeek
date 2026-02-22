#### How to solve KC cache memory problem?
1. Multi-Query Attention (MQA)
* What if all the attention heads share same Key & Value matrics?
3. Group-Query Attention (GQA)

***

* 15:00

* How? All attention heads can have the same K,V matrics.

| Attention Mechanism Type | Size of KV Cache | GPT-3 (175B, l=96, n=96) memory needed |
|---|---|---|
| Multi-Head Attention (MTA)  | l.b.n.h.s.2.2 | 4.5 GB |
| Group-Query Attention (GQA) | l.b.h.s.2.2   | 48 MB|


* DeepSeek has 128 attention heads!
* MQA reduces size of KV cache by 1/128 (400GB (MHA) -> 3GB (MQA))

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

* 30:00

***

* 35:00

if you see for the different heads the diversity for many of these heads is not that high they look almost kind of
35:10
similar this effect is not that pronounced in this visualization right now but this highlights one mean
35:16
disadvantage of multiquery attention and the disadvantage is that the diversity
35:23
among different attention HS is not that much so we don't capture as many perspectives so our model becomes weaker
35:29
our model does not perform as well all right so this brings us to the
35:35
end of this lecture on multiquery attention multiquery attention is the first method which people or researchers
35:41
invented to solve the KV cach memory problem and the idea is that we share
35:46
the same key matrices and value matrices across all the heads similarly we share
35:53
the same key weight matrices and value weight matrices across all the heads and
35:58
that reduces the KV cache by a size of n so for gpt3 if the KV cach took 4.5 GB
36:06
since we have 96 attention heads for if you use multiquery attention There is
36:11
almost a factor of 100 difference reduction in the amount of memory needed deep seek has 128 attention heads
36:19
so multiquery attention actually reduces the KV cach size by factor of 128 so from 400 GB we go to 3 GB that's the
36:27
good side of multiquery attention the Dark Side of multiquery attention is that it defeats the purpose of
36:33
multi-head attention the main purpose of mha was to capture different perspectives through different attention
36:39
heads in multiquery attention this purpose is defeated because well although different
36:45
attention heads value will be different because the query vectors are different across the different heads but the keys
36:51
and the value vectors are exactly the same so we cannot capture as much diversity so the model per performance
36:57
will degrade the language model will not be as good in capturing the complexity of the underlying sentences or the
37:05
paragraphs in the next lecture we are going to look at another method to solve the KV cach problem and that's the
37:11
grouped query attention we'll also see a code for the group query attention and
37:16
then finally we'll move to the multi-head latent attention which is the key innovation in the Deep seek paper
37:22
thanks a lot everyone I hope you are liking these lectures please make notes now we are getting deeper and deeper
37:28
into the deep seek Innovations so as we get deeper when you follow along things
37:34
might get a bit challenging so make notes you can share the notes with me ask doubts Etc thanks everyone I look
37:41
forward to seeing you in the next lecture
Multi-Query Att
