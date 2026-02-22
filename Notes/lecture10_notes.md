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

medium visualization so let's first see for head number zero this is the trainable Matrix for head number0 Z it's
30:29
the same thing as the first uh let me go over here it's the same thing as this
30:35
first first head trainable weight Matrix but let's see the dimensions here the
30:41
number of columns here are equal to 64 why so if you see 0 2 4 it goes up till
30:47
64 the number of columns are 64 because the head Dimension is 64 here the head Dimension was two but now this is just
30:53
64 in our example and the number of rows are 15 so the number of rows here are
30:59
actually equal to the number of embedding Dimension and the embedding Dimension is I think 1024 so we cannot
31:06
show all the one24 here so I have just shown 50 rows over here so that's why it's 50 rows and 64
31:12
columns 64 is the head Dimension and 50 is just a choice I made because I can't show the entire one24 rows so take a
31:20
look at this heat map so every color here represents one value this is the heat map for head number one heat map
31:27
for head number two you sorry head number zero and head number one you'll see that the heat map differs from head
31:32
zero and head one right it differs from head one to head two so you'll see that all the keys have different heat Maps
31:39
over here which is expected since the keys don't share the same values even the value matrices so now I'm plotting
31:45
the value Matrix for head zero value Matrix for head one for head two they they all seem to be very different from
31:52
each other now you'll see for the Falcon model this is the key Matrix which has
31:58
been plotted for head number zero which means that this is the
32:03
trainable key Matrix which is plotted for the first head right now and now if you scroll
32:09
down below you'll see that the values for head zero for head one for head two
32:15
for head three for head four for head five they are all the same because all of these heads now share the same key
32:22
values similarly if you go to the value matrices you'll see that for head number zero for head number one for head number
32:29
two for head number three for head number four and for head number five all of these heads share the same Val value
32:37
matrices why do they share the same value matrices because that's the main thing which is implemented in multiquery
32:42
attention the same weight matrices are shared across different heads that's the key advantage of uh multiquery attention
32:51
so I'm just visualizing it here for you this is the second visualization I wanted to show for gpt2 the values are
32:58
different across different heads and for Falcon model the values are same across the different
33:04
heads the last visualization is I want to show you the attention score matrices
33:10
so first you can see for multi-ad attention I'm just looking at the last Transformer block remember there are
33:16
multiple Transformer blocks um and if you see in each of these I'm showing
33:22
that so now if you see I have the quick brown fox the quick brown fox jumps over
33:28
the lazy dog right so there are nine tokens so the attention score Matrix will be 9 by9 let me write it on the
33:35
white board over here the quick brown fox jumps over the
33:42
lazy dog the quick brown fox jumps over the lazy dog so this is the attention
33:49
score Matrix for the first head uh similarly there will be an attention score Matrix
33:56
for the second head Etc I have just shown these values over here I have just shown these values over here for the
34:02
different uh for the different heads you see that's what this heat map actually
34:07
indicates so gpt2 has 16 heads I think gpt2 has 16 heads and you'll see this
34:13
attention score Heat Map for all the 16 heads you'll see that it's different from each other and then for Falcon I've
34:20
shown the same thing but Falcon model has actually 32 heads so for each head I
34:25
have shown the attention score Heat map which is 9 by9 because there are nine tokens and here what I actually want to
34:33
point out is that the main advantage of multi-head attention is that each head can capture a different perspective and
34:39
each head does capture a different perspective so ideally you should see more diversity in what all is captured
34:45
through these different attention heads whereas if you see the Falcon model right now it looks kind of similar but
34:51
overall the effect is that the diversity captured by each head is not that high because the keys and the values among
34:58
these different heads are shared so that's what I wanted to show through these plots that for the Falcon models
35:05
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









