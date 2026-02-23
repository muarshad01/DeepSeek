## Group Query Attention (GQA)

* Normal Multi-Head Attention (MHA) - 400 GB
* Multi Query Attentino (MQA) - 3 GB
* MQA reduces the KV size by a factor of 128

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
| Multi-Head Attention (MTA)  | l.b.n.h.s.2.2           | 4.5 GB            | 400 GB |
| Multi-Query Attention (MQA) | l.b.1.h.s.2.2 (n=1)     |  48 MB            |   3 GB |
| Group-Query Attention (GQA) | l.b.g.s.2.2  (n -> g)   | 384 MB (8 groups) |        |



| Attention Mechanism Type | Number of unique key-value (KV) pairs | KV cache size | Performance (Context Understanding)|
|---|---|---|---|
| MHA | | Largest  | Best |
| GQA | | Medium   | Medium |
| MQA | | Smallest | Worst |






matrix from group one and I need to cache one matrix from group two. So essentially based on the number of
20:47
groups which I have I need to cache those many matrices. If I have 10 groups
20:52
I cache 10 matrices one from each group. So this n factor here, this n factor here is now replaced
21:02
with g. So let's say in deep seat this n was 128 and if the number of groups are 8,
21:09
this is now replaced with 8. So there is still the kv cache reduction which we obtain 128 by 8. In the multi-query
21:18
attention the kv cache reduction was the maximum because it was 128 by 1 which is 128 times. In the case of uh group query
21:26
attention the reduction would be 16 times 128 divided by 8 which is 16. So
21:32
GQA has better performance than MQA. Uh so this is in terms of diversity. In
21:39
diversity when I mean diversity I mean the different perspectives which my attention heads can capture making my
21:45
language model better. GQA is better than MQA but it's still not as good as MHA. we are still doing a bit of
21:52
cheating. We are not doing as much cheating as we did in multiquery attention. But still slight amount of
21:58
cheating is being done. Um so if you actually compare multi head attention
22:04
versus multi-query attention versus grouped query attention, you'll have to compare along two verticals. First is
22:10
the KV cache size and one is the performance. In the KV cache size, multi head attention is the largest because it
22:17
takes all the attention heads. Multi-query attention is the smallest
22:22
because there is a division factor of 1 by n and group query attention lies
22:27
somewhere in the middle. It's still lesser than multi head attention but it's higher than multi-query attention.
22:33
So the factor here is actually g by n where g is the number of groups. That's for the kv cache size. For the
22:40
performance which is context understanding, multi head attention performs the best because every head has
22:47
different values of key and value matrices. So it captures more diversity, more perspectives. Multi-query attention
22:53
is the worst. The reason it's the worst is because uh um the reason is the worst is
23:01
because every head has the same value. And uh you'll see that grouped query
23:06
attention lies somewhere in the middle. Again, it lies somewhere in the middle because it's better than multi-query
23:13
attention, but it's not good good at good as multi head attention in terms of performance or context understanding.
23:20
So, you see grouped query attention lies somewhere in the middle. It's in terms of KV cache size is in the middle
23:26
between MHA and MQA. In terms of performance also, it's in the middle between MHA and
23:31
MQA. So, that's about group query attention. Once you have understood multi-query attention, group query
23:37
attention is just a small extension where we create groups. Now what I've done is that I've created a small code
23:44
which we are going to go through where we are going to take a very well-known model which uses group query attention
23:51
and that model is llama 38 billion. So if you see llama 3 8 billion and if you
23:57
search group query uh you'll see that meta themselves have
24:03
had mentioned when the llama 3 was launched uh on April 18, 2024 they
24:09
themselves have mentioned that they have adopted group query attention across both the 8 billion and 70 billion sizes.
24:15
So when I was constructing a tutorial for this topic I was going through models which I can use to explore.
24:22
Naturally I came across llama 3. So then I wanted to explore further with llama 3
24:27
and I wanted to actually create a heat map across the different heads so that I
24:32
can visualize the groups which are there and I wanted to see whether the keys across groups are the same and the
24:39
values across groups are the same. So that's what I did actually exactly here
24:44
is my hugging face key which should not be exposed but you can create you can
24:51
log into hugging face and to access this model through hugging face you'll need to submit the access request actually so
24:57
you can click on this and uh I have access to this right now but if you don't have access you'll need to submit
25:04
your details and only then you can get access once you get access in 1 to two
25:09
hours you will get access and a notice will come that you have been granted access to this model and then you can
25:15
start using it. So they require your details but I guess it's fine at least
25:20
they made the full thing open source. So then what I do is that I download this model from hugging face
25:26




***


that is the first thing which I do and it does not take too much time. Although I ran it on an A100 on a T4 GPU also
25:33
this will run. I ran it uh I downloaded it through hugging face and then what I did is that I wanted to visualize. So
25:40
first let's look at the keys only. Let's look at the keys. Let's worry about the values later. What I wanted to do was
25:47
that I wanted to create some sort of a visual like this. I wanted to look at
25:53
group number one. I wanted to look at group number two, group number three, uh
25:59
etc. So I had read somewhere that this model actually has eight groups. This
26:05
model has eight groups and each group has four headers. So there there are 32
26:10
total attention heads and each group has four heads. So group one has four attention heads, group two has four
26:17
attention heads etc. What I wanted to do was that I wanted to visualize
26:23
uh I wanted to visualize the query I wanted to visualize the keys um I wanted
26:29
to visualize the keys matrix and I wanted to visualize the values matrix. So this I wanted to visualize the keys
26:35
matrix for every group. So now um there will be K1, K2, K3 and K4 the keys
26:43
matrix for group one. Then for group number two there will be K5, K6, K7 and
26:49
K8 etc like that. Uh and what would be the dimensions of the keys matrix? The
26:55
dimensions of the keys matrix will be based on the input tokens which I have and uh the head dimension essentially.
27:02
So I'm looking at so if you look at this the number of rows will be the number of
27:07
input tokens and the number of columns would be the dimension for that
27:13
particular header. So I just want to visualize ideally K1 should be equal to
27:19
K2 should be equal to K3 should be equal to K4 because within one group all the key matrices will be the same but for
27:26
the second group the key matrices should be different from each other. That is what I wanted to visualize. And then I
27:32
just wrote a simple code for this where I took the keys uh I took the
27:38
keys matrix and then I just uh constructed a heat map. So for example,
27:43
if you see every row corresponds to a particular group the first row corresponds to group number zero. So
27:50
here the indexing is from zero group group one then group two and right up till group number seven. So if you look
27:56
within one particular group, if you look at group number zero, you'll see that for head 1, head two, head three and
28:03
head four, we have the same values. Look at this, this, this, and this. It's the same content because within one group,
28:10
all the matrices have to be the same. If you look at the second group, within that second group, you'll see that all
28:15
the matrices are the same. Similarly, if you go right till the very end and if you look at the last group, you'll see
28:21
that all the matrices visually are exactly the same. This is proof that Meta Lama 3 used the group query
28:28
architecture. And now if you see if you see if you compare group zero and group
28:34
one it's different. Group one and group two it's completely different. Group two and group three it's completely
28:39
different. So within one group it is the same. The key value the key matrices are
28:44
all the same but across different groups they are different. That's the advantage of GQA in terms of MQA.
28:51
uh especially in terms of uh the diversity which we discussed capturing
28:57
um more context doing well with respect to understanding perspectives
29:03
etc. So I hope this visual makes it clear to you that llama 38 billion uses
29:09
a grouped query attention because in each group you'll see that the key matrices are exactly the same. Then what
29:16
I did is I did the same thing for the values. So I did the same visualization for the values where for the values also
29:23
within one within let's say group number one v_sub_1 should be equal to v3 should
29:28
be equal to v v_sub1 v_sub2 v3 and v4 then in group number two v5 should be
29:35
equal to v6 should be equal to v7 should be equal to v8 this is exactly what I
29:40
tested next so my next plot was exactly replicating this but now I did it for
29:45
the values so if you see group zero has has the same values matrix across all my
29:51
heads. Head one, head two, head three and head four. Group number the next group has the same values matrix across
29:58
all the heads. Similarly, if you go right down till the very end, the last group has the same value matrix across
30:03
all the heads. But now if you compare between the first group and the second group, the value matrices differ. From
30:10
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










