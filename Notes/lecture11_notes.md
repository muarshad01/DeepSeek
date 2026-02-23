## Group Query Attention (GQA)

* Normal Multi-Head Attention (MHA) - 400 GB
* Multi Query Attentino (MQA) - 3 GB
* MQA reduces the KV size by a factor of 128 for DeepSeek

#### Disadvantages of MQA
* Significant performance degradation
* Remember the purpose of MHA: each head captures a different perspective!
* MQA: Diversity across multiple heads is very low!!!

***

* 10:00

1. MHA (ALL heads are different) - Complete diversity!
2. GQA
3. MQA (ALL heads are same) - Least diversity - performance degradation!

* GQA: Instead of all attention heads kaving same K-V matrices, what if we create groups of attention heads.

***

* 15:00

#### Diversity
* MHA >> GQA >> MQA

#### Memory
* MHA << GQA << MQA

***

* 20:00


| Attention Mechanism Type | Size of KV Cache | GPT-3 (175B, l=96, n=96) memory needed | DeepSeek (n=128)  memory needed|
|---|---|---|---|
| Multi-Head Attention (MHA)  | l.b.n.h.s.2.2           | 4.5 GB            | 400 GB |
| Multi-Query Attention (MQA) | l.b.1.h.s.2.2 (n=1)     |  48 MB            |   3 GB |
| Group-Query Attention (GQA) | l.b.g.s.2.2  (n -> g)   | 384 MB (8 groups) |        |

***

| Attention Mechanism Type | Number of unique key-value (KV) pairs | KV cache size | Performance (Context Understanding)|
|---|---|---|---|
| Multi-Head Attention (MHA)    | H (Each head has its own K and V)                              | Largest  | Best |
| Grouped-Query Attention (GQA) | 1 (ALL heads share the same K and V)                           | Medium   | Medium |
| Multi-Query Attention (MQA)   | G (Heads are divided into G groups, each group shared K and V) | Smallest | Worst |


#### [Introducing Meta Llama 3: The most capable openly available LLM to date](https://ai.meta.com/blog/meta-llama-3/)
* Llama 3 adopts grouped query attention (GQA) across both the 8B and 70B sizes.

***

* 25:00

*** 

* 30:00


the second group and the third group, the value matrices differ. So uh group
30:16
one has different value matrices than group two dot dot dot right up till the last group. This is exactly how group
30:23
query attention works. In one group all the keys matrices all the value matrices
30:28
will be the same. So the keys for head one will be the same for as keys for head two will be the same for keys for
30:35
head three and keys for head four. Values for head one will be the same as values for head two will be the same as
30:40
values for head three will be the same as values for head four. That's for one group. This will hold the same for all
30:46
the groups. But between groups, it will be different. For example, group one content will be completely different
30:52
from group two content. Will be completely different from group three content etc. So I hope through this
30:58
visualization you understood that latest state-of-the-art models are also implementing the architectures which we
31:04
are discussing in these lectures. It's just that it's very difficult to find such detailed explanations anywhere.
31:11
people have written blog posts but I find that such visualizations which I'm showing right now are very important for
31:17
your understanding and that's why I'm making these lectures a combination of whiteboard visualizations plus going in
31:24
depth through the code. I don't want you to understand the code per se but I plan to just show uh that you can download
31:32
models from hacking face and you can explore these models further once you understand these
31:37
concepts. So this is about grouped query attention. As a quick recap in grouped
31:43
query attention what we do is that instead of all the heads sharing the same keys and values uh we create groups
31:50
and within one group we have the same values. So let's say group one in group
31:55
one all the heads share the same keys and in group one all the heads share the same values. In group two all the heads
32:02
share the same keys. In group two all the heads share the same values. that is similar for the keys matrix, values
32:08
matrix and also W K and WV. So the advantage of grouped query attention or
32:14
multiquery attention is that we are not saying that all heads will share the same thing. We are creating groups. So
32:21
in terms of capturing diversity, it's better. Um and then there is also a
32:26
trade-off with respect to size. So if you compare grouped query attention versus multiquery attention in grouped
32:33
query attention um let's see this right this is a good good uh table to compare
32:39
if you compare grouped query and multiquery attention you'll see that in group query attention there are g heads
32:46
which need to be cached uh which are the number of groups whereas in multiquery
32:51
attention we just need to cach uh just nothing we just need to cach one head
32:56
but here we have to cach one head for every group. So the memory size increases by a factor of G from
33:04
multiquery attention to group query attention. That's the disadvantage of GQA over MQA. But the advantage is of
33:11
course the performance of GQA is better than MQA. And if you compare all of these three models together, multi head
33:17
attention versus multi-query versus group query, you'll see that in terms of performance context understanding, multi
33:24
head attention performs the best and multi-query performs the worst. Similarly, in terms of KV cache size,
33:30
multi head attention has the largest, KV cache size, MQA has the smallest and grouped query lies somewhere in the
33:36
middle. But if you the main conclusion of the lecture is that GQA tries to optimize
33:43
both. So in terms of performance and in terms of memory it lies in the middle between multi-query attention and multi
33:50
head attention. What deepse did is that they tried to answer the golden question. The
33:56
golden question is that can we create something else over here for which my KV cache size would
34:04
also be low and my performance would also be very good. Can I create something like this? Essentially I want
34:10
performance to be as good as the multi head attention and I want my KV cache to
34:16
also be very small. So can I have the best of both worlds and it looks like an impossible problem to solve right? If
34:23
you try to have a good performance, you will of course have to have more memory
34:28
and but they solve this problem by introducing a very beautiful trick which is the multi head latent attention. In
34:35
the next lecture, we are finally going to start looking at multi head latent attention in a lot of detail and uh
34:42
there is also a coding module which I have in which we'll try to implement the multi head latent attention from
34:47
scratch. I know we have been building up to this lecture for quite some time now but finally we are at this stage where
34:54
we have literally covered everything. We covered self attention, we covered causal attention, we covered multi head
35:00
attention, we covered key value cache, we covered MQA, we covered GQA. The only
35:05
thing which is now remaining is to finally tackle the key innovation in the deepseek architecture. That's the multi
35:11
head latent attention. This is what we'll be seeing in the next lecture. As the lecture gets more lectures get more
35:17
deep and technical, I encourage you to maintain notes, maintain detailed notes of what you are learning so that you
35:24
don't feel lost along the way. And until now it was very important to explain all these concepts to you because otherwise
35:31
it will be very hard to understand latent attention and the beauty of what deepse did they really got the best of
35:38
both worlds. They reduced the memory requirements of the KV cache and they also got a great
35:45
performance out of this mechanism called multi head latent attention which we'll look at in the next lecture. So thanks
35:52
everyone and I look forward to seeing you in the next lecture.















