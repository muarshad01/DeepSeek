#### Multi-Head Latent Attention

* How DeepSeek Changed Attention?

* [DeepSeek-V2: A Strong, Economical, and Efficient
Mixture-of-Experts Language Model - 2024](https://arxiv.org/pdf/2405.04434)

1. Key-Value (KV) cache
2. Multi-Head Attention (MHA)
3. Multi-Query Attention (MQA)
4. Grouped-Query Attention (GQA)
5. Multi-Head Latent Attention (MLA)

***

* 5:00

#### Key-Value Cache
* __Takeaway-1__: We seem to be doing a lot of repeated calculations.

***

* 10:00


layers, we get this logits matrix, right? And then the logits matrix
11:14
essentially is a vector for all of these tokens. So we see the logits vector for
11:19
is which is the last token in my input sequence and then we find the token ID
11:24
corresponding to the maximum probability that's going to be my next token. In this case the next token predicted is
11:31
bright. Now this next token bride is appended to my input matrix and my new
11:37
input matrix now becomes this which has five tokens. Correct? As I told you, the token which
11:45
is generated is appended back to the input sequence and it goes back through to the inference pipeline. Now let's say
11:51
I have this input sequence and again it goes through this same mechanism, right? Because this input new input sequence is
11:58
again passed through this entire inference pipeline. So then this new input sequence is again multiplied by WQ wkv.
12:07
We get the queries keys values. We multiply queries with the keys transpose to get the attention scores. We get the
12:13
attention weights and we get the context matrix. Remember now the context matrix is 5x4 because we have five tokens here
12:21
corresponding to the next day is bright. Now what I want you to do is I want you
12:27
to focus on these black boxes which I have marked over here and let me mark
12:32
them again in a sharper manner. So these are the black boxes which I have marked here. Right? this black box. There is a
12:39
black box here. Um, there is a black box here. There is a black box here. What do
12:45
these black boxes represent? These black boxes actually represent that we have already done these calculations in the
12:51
previous step. Here we calculated a 4x4 queries, keys and the values matrix.
12:58
This is the exact same 4x4 which is shown in this black box.
13:05
uh then this attention score we already calculated a 4x4 attention score in the
13:10
previous iteration right so here we are repeating this calculation again similarly in the attention weights even
13:16
in the context matrix we already calculated the context vector for the previous four tokens right so here we
13:22
seem to be repeating a huge number of calculations so everything shown in the
13:28
black box is actually repeated um so that's the first second takeaway
13:33
the first takeaway was we seem to be doing a lot of repeated calculations. The second takeaway is that yeah indeed
13:41
we are actually repeating everything which we have calculated before including the
13:46
queries, keys values, the attention scores, the attention weights and even the context
13:52
matrix. Then let us try to think from first principles that what do we really
13:57
need to do to predict the next token. So if the input sequence is the next day is
14:02
bright, what do I need to predict the next token? Let's say this is my input
14:07
sequence which passes through the LLM architecture. Right? What happens is that when this input sequence comes at
14:13
the right at the very end, we have this logits matrix, right? We have the logits matrix right at the very end over
14:20
here. Um let me mark it with the yellow color. Yeah, this is the logits matrix.
14:25
What does the logits matrix look like? It looks something like this. For all the for all my input tokens, the
14:32
next day is bright. I have a logit vector with size equal to my vocabulary size. And to predict the next token, I
14:39
only look at the logits vector for bright and we look at that ID which corresponds to the maximum probability
14:46
and then that helps me predict the next token. The main realization here is that
14:52
we don't need these other logits vectors at all. we only need the logits vector for
14:58
bride. That actually means that if we only need the logits vector for bride,
15:04
we only need the context vector corresponding to brite which comes out of the multiad attention block out. When
15:11
we come out of the multiad attention block, we get the context vector for all the tokens, right? But now since we only
15:17
need the logit vector for bride, to get the logit vector for bride, we only need the context vector for bride. So if you
15:25
see the context vector matrix which we had calculated in the previous iteration, we don't need all these
15:30
values at all. We only need this last row. Why? Because after the multi
15:36
attention block, the tokens are on their own. They don't care about each other anyways. So if I get the logit, if I get
15:43
the context vector for bride, that context vector will traverse through this rest of the journey and that
15:49
context vector itself will give me the logit vector for bride. So I don't need the context vector for the next day and
15:57
his. This is the third takeaway from our story. Our third takeaway is that to get
16:03
the next token prediction to get the next token prediction, we only need the
16:08
context vector for the last token in my input sequence. Keep this in mind. We
16:14
only need the context vector for the last token in my input sequence. Okay, that's a key realization.
16:23
So the question now we ask is what if we store the keys and the value matrices during
16:28
inference. What if we store the keys and the values matrix matrix during
16:33
inference? Will that help us to predict this last context vector matrix? Remember now we only need the context
16:40
vector for the last token. So if we only need the context vector for the last token, let us
16:46
backtrack and check what we really need. So if we need the context vector for my last token, I of course need the
16:52
attention weights for my last token and I need the values matrix. But what if I
16:58
cache the previous value matrix? What if I cach the previous value matrix and I
17:03
only compute the value vector corresponding to the new token? The way I'm going to do that is I'm going to
17:09
take the input embedding only for the new token and I'm going to multiply it by we this is anyway fixed during
17:17
pre-training. So I get the value vector corresponding to the new token and I append it to the cache. That's how I get
17:23
the entire value matrix. To get the attention weights for bride, I first need the attention scores
17:29
for bride. And to get the attention scores for bride, I need to multiply the
17:34
query vector for brite multiplied by the keys transpose. Similarly to what we did for
17:41
values, we can cache everything shown in the black box. And we need to only
17:46
compute the keys vector for my new token which is brite. So I take the input embedding for brite and I multiply it
17:52
with W K which is also fixed during pre-training. That's how I get this new keys vector. I append it to the previous
17:59
previous cache and I get my keys matrix. Similarly to get the query vector for brite I simply need to multiply the
18:06
input embedding for brite multiplied with WQ. That's all. So if I zoom out a bit here, you'll see
18:13
that the elements which I have marked in the boxes number one, number two and number three. These are the only three
18:20
computations which we need to do for any new inference. Which means that whenever any new inference whenever any new token
18:27
comes in, we first compute its query vector, we compute its key vector, we compute its value vector, we then append
18:34
the key vector with the previous cache to get the keys matrix. We append the value vector with the previous cache to
18:40
get the value matrix. Then we multiply the query vector with the keys transpose to get the attention scores. We get the
18:47
attention weights. The attention weights are multiplied with the value matrix. And then we get the context vector for
18:53
my new token. We only need this context vector to predict the next token. So
18:58
that's all we need. This actually proves that if we can store these keys and the values, if we
19:06
can cache the keys and the values, it will save us a lot of computations. We don't need to recomputee these values
19:12
every single time. And we don't need to cache the query query matrix at all
19:18
because we only need the query vector for the new token anyways. We don't need the entire query matrix. So we only need
19:24
to cache the keys and the values matrices. So the conclusion after this
19:30
demonstration was that the first major conclusion is that we need to cache the keys and the value matrices during
19:36
inference. This is called as the KV cache. Now the KV cache has a lot of
19:42
advantages. One major advantage of the KV cache being that if you plot the amount of computations needed versus the
19:48
number of input tokens. If you don't have a KV cache, the amount of computations needed increase
19:54
quadratically because you're repeating the calculations each time, right? But if you have a KV cache, the amount of
20:00
computations which you need don't grow quadratically with the number of input tokens. They only grow
20:07
linearly. Which means that even if you're doing more and more and more and more inference, since you're caching,
20:14
you are not increasing the cost quadratically. You are only increasing linearly. So you actually save uh so you
20:22
actually save this much amount of recomputations and that speeds up inference which is actually good. We
20:28
speed up inference due to the KV cache and it speeds up inference by a huge
20:35
magnitude. But what is the dark side of the KV cache? The KV cache actually comes with a dark side and that dark
20:41
side is that it takes up memory. Now remember these black boxes which I mentioned over here. We need to cache
20:48
them which means we need to store them in the memory. Right? What are the size? What is the size of this black box? We
20:54
have to have the number of tokens which is the um context size multiplied by the
21:02
dimensions. Right? So if you take a look at this black box we have to have the so
21:08
I'm calculating how many parameters we need to store in memory. So if the context size is denoted by s we need the
21:15
uh context size s multiplied by this dimension. So if my number of attention heads are n and if each head dimension
21:22
is h this will be n into h. So s into n into h number of parameters need to be
21:28
saved and if I have b batches then it will be n into s into h into
21:33
b and then this will be for the keys as well as the values. So this will be
21:39
multiplied by two. So if you take a closer look at the size of the KV cache formula, you'll see that it depends on
21:46
the context length which is number of my token my number of tokens in the keys and the values matrices multiplied by n
21:53
into h which is the keys and the values dimension number of attention heads multiplied by the head dimension
21:59
multiplied by b which is the batch size multiplied by two. So this first two is because I have k and v. there are two
22:06
caches and the second two is because I'm actually calculating number of parameters right
22:12
so these are number of bytes per floating point I'm assuming each parameter I'm assuming I'm calculating
22:19
the memory of each parameter and I'm assuming each parameter to be floating point to be a 16 bit floating point
22:26
which is two bytes so every parameter will take two bytes in memory that's why there is the second loop what is this L
22:34
this L is essentially the number of transformer blocks. This transformer block which I showed to you over here,
22:40
every transformer block will have a key value cache. And usually language models have 12, 24, 96
22:47
or even higher number of transformer blocks. So the size of the KV cache also depends on L which is the number of
22:53
transformer blocks. Now if you have a deepsek R1 model or V3, the number of
22:59
transformer blocks they have is 61. The batch size is one. The number of
23:04
attention heads which they have is 128 and the dimension of each attention head is
23:10
128. The context window is actually 100,000. So if you use this formula to
23:16
find the size of the KV cache, you'll see that the size of the KV cache for deepse R1 or V3 model is 400 GB. That's
23:24
a huge huge size. And why don't we want such a huge size? Because the higher the
23:30
size the more our memory will be overloaded. And if let's say if I am deepse and I have to store this KV cache
23:38
in memory which means I have to pay for this memory right it's like occupying land for every
23:44
piece of data stored we have to pay rent so deepseek has to pay that much amount
23:49
so then it will charge it in inference but the inference cost which deepseek charges is not that high at all which
23:56
clearly shows that they are not using this simplified version of KV cache they are doing a whole lot of new
24:02
modifications after this point which we are going to learn in today's lecture. But this is the dark side of the KV
24:08
cache. It takes up a huge amount of memory. Takes up a huge amount of memory. So it costs more. It also it
24:14
also slows down our other computations because our memory is overloaded. Imagine what happens if your laptop is
24:20
filled with memory, right? Your your tabs become slower. When you are trying to run a piece of code, it runs slow.
24:26
Your new windows open a bit slow. That's exactly what happens during inference time if you take up a lot of memory.
24:33
So these are the disadvantages of the KV cache. Now all of the innovations which
24:39
happened in the attention mechanism which I'm going to discuss in the rest of this lecture are there to solve these
24:45
disadvantages. So after this point people started thinking that okay the KV I of course need the KV KV cache because
24:52
I want to save these much amount of computations but I need the KV cache but
24:57
at the same time I want to solve the KV cache memory problem.
25:02
and it's clearly not being solved with the multi head attention mechanism. So then people started coming up with new
25:08
ways of solving the KV cache memory problem. Let's discuss these new ways
25:13
next. There are essentially two two major things people try to solve the KV
25:19
cache problem, KV cache memory problem. The first is the multi-query attention and the second is the grouped query
25:25
attention. These are the same diagrams which you'll see over here. the group query attention and the multi-query
25:31
attention. It's very difficult to understand anything from these diagrams. So I've simplified it a bit further
25:36
here. Let's say if you look at the normal multi head attention, right? What happens is that if this is W K and this
25:43
is WV, these are the weight drainable weight matrices for keys and the values.
25:49
You'll see that these are split into different heads, right? based on the number of attention heads which we have
25:55
and each attention head has different values because every attention head captures something different in my
26:01
underlying piece of data or underlying piece of text. So naturally when the input is now multiplied with W K and WV
26:08
we have the keys matrix and the values matrix. Now even when you look at the keys and the values matrix each
26:15
attention head has different values which are shown by different colors here. I've deliberately shown them by
26:20
different colors because every attention head has different values naturally. Now since every attention
26:26
head has different values, we need to store them separately in the cache. We need to store this head values in the
26:32
cache, this head values in the cache, this head values in the cache and this head values in the cache. And as a
26:38
result, if you see the formula for the KV cache, the formula for the KV cache depends on the number of attention heads
26:46
because every attention head has different values. So we need to store the keys and the values corresponding to
26:52
each attention head. Now these different colors is
26:57
exactly what the MQA and GQA target. The MQA and GQA say that instead of every
27:03
attention heads having different colors or having different values, what if attention heads could share the same
27:10
content then we would need to cache less, right? So that would reduce our memory requirement. Let me explain what
27:18
this means. First, let's look at multi-query attention. What multi-query attention says is that I'm not going to
27:23
touch my queries matrix at all. So, every attention head will have different Q1 value. Q2, Q3 and Q4. But what I will
27:32
do is that I will make sure that W K1 is equal to W K2 is equal to W K3 is equal
27:38
to W K4. And consequently, K1 is equal to K2 equal to K3 equal to K4. I'll also
27:45
make sure that wub equal to w2 equal to w3 equal to w4 and subsequently v_sub_1
27:52
equal to v_sub_2= to v3 equal to v4. What this means is that every attention
27:57
head will now share the same values. Every attention head will share the same key value. So k1 this is k1 this is k1
28:04
this is also k1 and this is also k1. That's why I shown it with the same color. Similarly for the values matrix,
28:11
every attention head will share the same content as V_sub1. This head one will be V_sub1. Head two content will be V1.
28:18
Head three content will be V1. Head four content will be V1. What this will do naturally is that since the content of
28:24
all the attention heads is the same. I don't need to cache all my attention heads separately. Right? I only need to
28:30
cache one head values. For the keys matrix, I only need to cach K1. And for
28:36
the values matrix, I only need to cach V1. that suddenly reduces my KV cache
28:41
memory size by a factor of uh by a factor of n. So the attention
28:48
head size actually this h should remain this h should this h remains as it is but here there was this n right which is
28:55
the number of attention heads that is now suddenly reduced. So I get a reduction of 1 by n in the size of the
29:01
kv cache because now I don't need to cache all my n attention heads in memory. I only cach one attention head.
29:08
Similarly, what is done in group query attention is that group query attention says that instead of making all the
29:13
heads to be the same, what if I make groups? So, for example, I make
29:19
head 1 and head 2 into one group and make all their values to be same. So, w k1 will be equal to w k2 and I have a
29:27
group two. So, w k3 will be equal to w k4. So, all of these won't be equal to each other. Only w k1 will be equal to w
29:34
k2. W K3 will be equal to W K4. Similarly, here only K1 will be equal to
29:39
K2 and K3 will be equal to K4 which you'll see from the colors. So, one group has one color. They share the same
29:45
content. Similarly, for the values W1 will be equal to W2 and W3 will be equal
29:52
to W4. That will translate to V1 being equal to V_sub_2 and V3 being equal to
29:58
V4. What this will do is that this will also reduce the number of uh uh the amount of
30:07
information I store in the memory right because now in each group I just need to store one attention head
30:14
information. So basically I just instead of a factor of n instead of a factor of
30:20
n now this n is replaced by a factor of g. So this should be g and this should be h. Now this n is replaced with a
30:27
factor of g over here because instead of saving all n attention heads, I only save for groups. So if I have 96
30:35
attention heads and only 16 groups, this will be 16 instead of 96 over here. So
30:41
you see the group query attention lies somewhere in the middle of multi-query and multi head attention. In multiquery
30:47
attention, all the heads share the same content. In grouped query attention, groups of head will share the same
30:52
content. Now naturally this will have consequences right this will have consequences for the size of the KV
30:59
cache. If you look at multi head attention it has the largest KV cache. So if you compare for GPT3 it occupies
31:06
4.5GB. If you look at the multiquery attention it has the smallest KV cache because I reduce by factor of n which is
31:13
96. GPT3 largest model has 96 heads. So the KV cache memory size reduces by a
31:19
factor of 100. And if you look at group query attention and if you assume eight groups, the size occupied by the group
31:26
query attention lies somewhere in the middle of MQA and MHA, it's about 384
31:31
mgabytes. So multi-query attention seems to solve this problem of memory, right? So you might be thinking then why do we
31:38
even need latent attention? Why not use multi-query attention? But this comes with a very big cost and that cost is
31:44
with respect to the performance of the language model. The reason we started with multi head attention is because we
31:50
wanted different heads to capture different diversity in my underlying data. But now if you share same content
31:58
across different heads, I'm I will not be able to capture enough diversity in my underlying data because all the heads
32:05
are now having the same keys and the values content anyways. So both the multi-query
32:10
attention and group query attention actually suffer in performance and they suffer in context understanding. So if
32:17
you look at this now if you compare all of these different mechanisms you'll see that multi head attention has the
32:23
largest KV cache size and it has the best performance whereas multi-query attention has the smallest KV cache size
32:29
but it has the worst performance because all values are shared whereas grouped query attention has a medium KV cache
32:35
size so it lies somewhere in the middle and group query attention also has a medium performance it lies somewhere in
32:42
the middle so multi attention has a great performance but it does not
32:48
solve the KV cache problem. Multi-query attention has solves the KV cache problem but it has the worst
32:54
performance. Even group query attention does not have that good of a performance. So then that's why people
33:01
thought can we get the best of both worlds which means that can I have a low cache size can I have a low cache size
33:09
like multiquery attention but can I have a good language model performance? Can
33:14
we get the best of both worlds? And when people started answering this question,
33:19
that's where the multi head latent attention was born. Let's learn about this
33:25
now. So how do we get the best of both worlds? First, let's quantify what best of both worlds actually mean. We want a
33:32
low cache size, right? Which means that in the size of the KV cache formula, these two terms which are there n into h
33:40
which are the number of attention heads multiplied by the head dimension. This seems to be in our control. Other things
33:46
like number of transformer blocks, the batch size, the context size. Let's say we do not play with these parameters.
33:53
But what if we take these and we make sure that this needs to be reduced. This n into h needs to be reduced for
34:00
deepsek. It's 128 into 128. That's a huge factor. If we can reduce this
34:05
somehow, that will reduce my KV cache size memory problem. Deepc currently takes 400 GB with normal
34:13
KV cache as we saw over here. Uh if we just implement the normal KV cache, it
34:19
takes 400 GB because it has 128 N into 128H. If we can reduce this
34:25
multiplication further, we'll reduce size of the KV cache. That's number one. Number two is that to have a good
34:31
language performance, what we need is that all the attention heads should have different values in the KV matrix. I
34:38
don't want to do something like this where I'm sharing values across attention heads because that seems to
34:43
reduce performance a lot. So I want different colors. I want k1 to be different from k2 to be different from
34:50
k3 to be different from k4. I want v_sub_1 to be different from v_sub_2 to be different from v_sub_3 to be
34:56
different from v4. That will only happen if this weight matrices this trainable weight matrices W K1 W K2 W K3 W K4 are
35:05
different from each other and W V1 W V2 V3 and V4 are different. How do I
35:11
achieve both these things? Because it seems that to reduce this I need to share some content over here. So it
35:17
seems impossible to achieve both of these things, right? How MQA reduced the size of the KV cache by making all these
35:24
colors the same. But how can we do this by having different values in KV matrix?
35:29
How can we reduce the size of the KV cache while still maintain good language model
35:35
performance? That's what DeepSync achieved with the multi head latent retention. And it's a beautiful thing
35:42
what they did to answer this question. But to answer this question, they asked a simpler
35:48
question. They thought about it differently. The way they thought about it was uh what if we don't have to cache
35:55
the keys and the values separately. So currently what's happening is that the keys and the values are being cached
36:01
separately right and this n into h factor is coming that's being multiplied by two. What if we don't cache the keys
36:08
and the values separately. What if we cache only one matrix and furthermore
36:13
what if this matrix has lesser number of dimensions than n into h. So currently
36:18
we have this 2 into n into h right two because we are caching k and v separately and n into h because that's
36:26
the dimension of the keys and the values. What deepse asked instead was
36:32
what if we don't have this factor of two instead we just have one matrix and instead of n into h we have
36:39
something else we have another dimension which is much lesser than n into h. Can
36:44
we do this? Can we cache only one matrix instead of two? And can we make sure
36:49
that this matrix has less dimensions? So this seems like something like a magic
36:54
bullet, right? Or a magic formula. How is this ever going to happen? The way they did this is that
37:01
they used the concept of a latent space. The idea of latent spaces has been used in machine learning and deep learning
37:08
for a very long period of time. But it's the first time they applied it to language modeling. So to get this matrix
37:15
which has less dimensions than n into h and if you only need to cache one matrix
37:21
they said that to get this matrix we start by projecting the input embedding matrix into a latent
37:27
space. So let's say this is my input embedding matrix right the next day is
37:32
I'm first going to show you how the uh latent KV cache mechanism works or how
37:40
the latent attention mechanism is implemented and then I'm going to walk you step by step through it. So first
37:45
they started by saying that what if we project the input embedding matrix into a latent
37:51
space and the latent space what does latent space mean? First of all, the input embedding matrix is a size of 4x8.
37:59
What they said is that what if we multiply it with this w dv. D means down
38:06
projection and kv means the kv cache. For now uh it's fine if you don't
38:12
understand this but let's just understand what we are doing here. Here we are multiplying the input embedding
38:17
matrix with another matrix and then we have the CKV matrix. This is that matrix
38:23
which I was talking about. What deepseek showed was instead of caching the keys and the values they showed that we only
38:29
need to cache this matrix and then we will achieve the best of both worlds. They said that this is our latent matrix
38:37
which we will cache and that will achieve the best of both worlds. That
38:42
will help us reduce the size of the KV cache and we won't even need to share values among
38:47
heads. Now you might be confused that what are we exactly doing here? We seem to be introducing one more matrix. How
38:54
is this actually helping us? So let me first show you the full workflow of what they did. Actually the first step is the
39:01
only place where some changes are happening because here we introduce this latent space right. So this is now my
39:07
latent matrix. This will be multiplied with wuk and
39:12
wuv. U is basically a projection. So now with this multiplication we are
39:18
converting it into into the query matrix and the value matrix. For the sorry the
39:23
keys matrix and the values matrix. For the queries it stays the same. For the queries we are going to take the input
39:29
embedding matrix we are going to multiply it with the WQ to get my queries matrix. Later versions of latent
39:35
attention are a bit more complex where the queries are also projected to a different space. But for now I'm
39:41
projecting the queries the same way as we did before. The queries are multiplied with W K. The sorry the input
39:47
embedding matrix is multiplied with WQ and we get the queries matrix. X multiplied by WQ gives me the queries
39:54
matrix. The change happens to get the keys and the values matrix. Now we don't
40:00
get the keys and the values matrix through through W K and WV. Instead what we are doing is that we are having this
40:07
latent matrix which is CV. We are multiplying C KV with W UK and W UV.
40:13
That's how we get the keys matrix and the values matrix. After this, everything stays the same. The queries
40:19
are multiplied with keys transpose to get the attention scores. We get the attention weights. The attention weights
40:25
are multiplied by values to get the context matrix. So then you must be thinking what really changed over here,
40:32
right? Because in the earlier case also we were multiplying with W K and WV
40:37
instead of this W U K and W U. But here we seem to be introducing an additional
40:43
step. So instead of reducing my KV cache, I seem to be increasing my KV cache size. U but that's where we'll see
40:53
how does adding this latent matrix actually help. And what deepse call is that they call this trick of adding this
40:59
latent matrix an absorption trick. So here we'll need to go into some amount of mathematics to see how addition of
41:06
this latent matrix actually helps. And once you start seeing this, it's actually quite beautiful. So let's first
41:12
start with what we are doing here, right? We are starting with the input embedding matrix. We are multiplying
41:17
with it with WQ to get my queries matrix. That's step number one. Done.
41:23
Then what we are doing is that we are multiplying my input embedding matrix with this WDKV and we are getting my
41:29
latent matrix. That's step number two. Done. Then we multiply this CKV with W
41:34
UK to get my keys matrix. And I multiply this CKV with Wuv to get my values
41:40
matrix. So CKV * W UK is just X multiplied by WDKV multiplied by WUK.
41:49
And CV * W UV is just X multiplied by WDKV multiplied by
41:56
Wuv. Now let's see what this absorption trick is. Eventually you'll want to find the attention scores, right? So you'll
42:03
need to find queries multiplied by the keys transpose. Now queries is X into WQ
42:09
and keys transpose will be transpose of this which will be W UK transpose W DKV
42:15
transpose and XR transpose. Here is where the absorption actually comes into the picture. What I'm going to do now is
42:21
I'm going to absorb these two together. I'm going to absorb
42:28
WQ and W together. So this same multiplication can be
42:34
written as X into WQ into W UK transpose multiplied by X into WDKV transpose.
42:42
Okay. So now these two matrices if you have a closer look at it WQ and W UK
42:49
transpose these two are fixed at training time. What this means is that we have already all these trainable
42:56
weight matrices they are fixed during pre-training right we don't need to compute them during inference. So these
43:01
two are fixed during training time. So we don't need to store any store this at all. This will not cost us anything.
43:07
This is already fixed. We already compute them during pre-training. Now if you carefully observe this quantity,
43:14
this quantity is nothing but my CKV and we can cache this
43:22
quantity. So to get my attention scores now what I can simply do is that I can
43:27
find X. So if a new token comes in I just have to find this absorbed query
43:34
what is my absorb query? My absorbed query is just the new token multiplied by this vector this whole matrix that
43:41
gives me my absorb query and then I have to just multiply it with the cache this
43:47
latent cache. So this matrix is the only one which will be stored in the cache and I just need to multiply with the
43:54
cache value. This is what is called as the absorption trick. Through the absorption trick,
43:59
they showed that addition of this matrix actually helps us because WQ and Wuk can
44:05
be absorbed together. So we already know this. So even if we only cached this, it
44:10
helps us compute the attention scores. Now if you move to the attention the
44:16
context vector, you'll multiply the attention weights with the value matrix, right? So you'll multiply the attention
44:22
weights which is Q into K transpose multiplied by V. And what is V? V is
44:27
just X into W DKV into W UV. So V is X into W DKV into W
44:33
UV. So this is my context vector matrix and then we'll ultimately get the logits
44:39
matrix by multiplying it with some output projection head. So similarly here we can absorb this W UV and the
44:46
output projection head into one into one bracket and this only needs to be computed once during pre-training. This
44:53
is fixed and this can be cached similarly. So what I'm trying to show here is that
45:00
if we cache CKV we can compute the attention scores and we can compute the
45:05
context vector matrix which will help us compute the next token also. So this will help this this is the formula for
45:11
the context vector matrix and this is the formula for the logits matrix which helps us compute the next token. So I'm
45:18
I'm showing here that even if we cache only one matrix it will help us go through this entire pipeline even if we
45:24
cache this latent matrix it will help us compute the attention scores and it will help us compute the next token also. So
45:32
you might be wondering that this sounds a bit abstract and I'm not fully being able to understand what's the flow here.
45:39
So for that what I've done is that I have now I'll show you a separate section where I'll show you visually
45:45
what happens when a new token comes in and how this absorption trick actually helps us. For now just remember that
45:53
what we are actually going to cash is X into WDKV which means that X multiplied
45:59
by WDKV if if this is the result this we are going to cash again I'm shown this
46:06
with black box remember this these black boxes I was showing you initially separately for the
46:12
uh I was showing you these black boxes for caching the values and the keys separately right but here I'm showing
46:19
that instead of caching the keys and the value separately. We only need to cache CV which is the latent matrix multiplied
46:27
which we get after multiplying X with WDKV. So if this is sounding a bit
46:33
abstract let's see what happens when a new token comes in. So if a new token
46:38
comes in let's say my new token is again bright similar to what we saw before. If
46:44
a new token comes in, as I mentioned, what we need to first do is we need to find the absorbed query, which means
46:51
that we just need to get the X for this new token and we need to multiply it with WQ multiplied by W UK transpose. So
46:59
I get my new my I get my new input embedding for bright and I multiply it
47:05
with WQ multiplied by W UK transpose. Compare and contrast this with what we
47:10
did in the actual key value cache. In the actual key value cache, we just multiplied it with WQ. But now we have
47:17
absorbed W UK transpose within this. So whenever a new token comes in, I just
47:23
multiply the input embedding for the new token with WQ and WK transpose. Nothing
47:29
here needs to be cached because we already fixed these things during pre-training. So I get my new query
47:35
vector done. This is my absorbed query vector for Bright. Absorbed query vector for Bright. Then what I have to do to
47:42
get the attention score is I have to calculate my updated cache. Okay. So to get my updated cache I have to first
47:48
find the new cache vector new KV vector. So my new KV vector is my input
47:55
embedding for bride multiplied with this WDKV and that gives me my latent
48:00
embedding for my new vector which is bright. I append this embedding to my KV
48:05
cache. So this was my previous CKV and this is my updated or appended vector to
48:11
this CKV. This is my updated KV cache. So what I will simply now do is
48:17
that I will multiply this query vector which absorbed query vector with this updated KV cache. That gives me my
48:24
attention scores for bright. Then I get my attention weights for bright. Then
48:30
what I'm going to do is that uh I'll need to get this updated KV cache and
48:36
multiply it with W UV to get my values matrix. Um and so see this updated KV
48:44
cache is shared for my keys and the values. I don't need to calculate a separate value cache here. I use the
48:50
same updated KV cache which we had computed here for the uh attention scores computation and I use it for the
48:57
values matrix computation. I multiply this with Wuv. Again, this is fixed after pre-training and I get my values
49:03
matrix. Then I will multiply the attention weights for bright multiplied with the values matrix and then I get my
49:10
context vector for bright. Remember I only need the context vector for bright to make the next token
49:17
prediction. This is awesome, right? Uh because what we are doing here is that I
49:22
have just showed you what happens when a new token comes in. First let us see earlier what we did when a new token
49:29
came in earlier what we did when a new token came in. When a new token came in earlier remember what we did first we
49:36
found the query vector we found the key vector we found the value vector that much is fine but then we need two caches
49:42
earlier we need the keys cache and we need the KV cache and we need the Vcash. Sorry we need the keys cache and we need
49:49
the values cache. But now you see we don't need these key cache. We don't need the key
49:56
cache and we don't need the value cache. We just need one KV cache or one latent cache which is this CKV. We just need
50:03
this one latent cache which is this CKV. So where is this latent cache used? I
50:08
use this latent cache to get my updated uh so I need to compute the attention
50:15
scores right. So for here I need my cache to get the um attention scores. To
50:21
get the attention scores, I multiply my absorbed query with my updated KV cache.
50:27
And then I get my attention scores. From this, I get the attention weights by scaling and applying causality. And then
50:34
I need the updated KV cache once more to multiply with Wuv to get my values
50:39
matrix. Of course, this Wuv will later be absorbed with Wo to get my logits
50:45
vector. Don't worry about that for now. For now, just think about the fact that we need this updated KV cache just once
50:53
and that is shared to get the attention scores. That same KV cache is used to get my context vector because I use this
50:59
KV cache to get the values matrix and then the attention weights for the new token is just multiplied with this value
51:05
matrix to get the context vector for bright. This is exactly how multihead
51:10
latent attention actually works. And then we can see that did we actually solve two problems which we started
51:17
with the so let's see right what are we storing here in memory what we are storing here in memory is just my
51:25
um what I'm storing in memory is just my latent KV matrix right which is cv and
51:31
the size of this is again given by s which is my context s which is the context size but
51:38
the main reduction is now obtained with the number of columns the number of columns here which was n into h before
51:44
right for the keys and the values matrix now it is not n into h now it is just the dimensions which I have to choose
51:51
for my latent matrix so I have to choose these dimensions right currently I choose the dimensions of my latent
51:58
matrix to be uh I chose my dimensions to be four dimension so already we are
52:04
going from eight dimensions to four dimensions I can choose it to be anything so instead of n into h now that
52:10
is given by n ss And actually earlier we were computing two two caches right K and V. So this
52:16
two factor is also now gone here. So let's see the size of the latent KV cache. Earlier the size was L into B
52:23
into N into H into S into 2 into 2. Now 1 2 will be gone because we are not
52:29
saving K and V separate caches. We just need one cache. And instead of N into H now I have this DL which is the
52:35
dimension of the KV cache. So for the deepseek paper now if you see
52:41
deepsek uh if you use the direct multi head attention they have they were getting I think
52:47
uh 40 uh 400 GB of space but what deepseek
52:54
actually did was in multi head latent attention they used a DL of I think
52:59
uh what was the DL which they used they used a DL of 576 I think they used a DL
53:06
of 576 six. So the reduction which you get for deepsek was is 2 into n into h
53:13
which was for m the original voltage attention divided by dl which is 576. So
53:19
you get a 57 times reduction in the size of the kv cache. So instead of now
53:24
saving 400 GB you you reduce this by 60 which is almost 6 GB now which is incredible. So we definitely get a size
53:32
reduction. We definitely get a memory size reduction. Earlier we were do calculating 2 into n into h but now this
53:40
two is gone and instead of n into h for deepsek n into h was 128 x 128. So it
53:46
was 2 into 128 by 128 but now it's not like that. Now we just have 576. So the
53:52
reduction in size which we get is 2 into 128 into 128 divided by 576 which is by
53:58
57 times. So it definitely solves the first issue which is it reduces the size
54:04
of the KV cache by 57. Does it solve the second issue of uh
54:10
sharing content across different heads? So if you carefully watch what we did over here, the beauty of latent
54:16
attention is that WUK Wuv they have different content across each head. So
54:22
this keys matrix and the values matrix has different content across each heads. So we are not sharing the same content
54:29
across different heads at all. Every head will now have different content for the keys and the values. This is exactly
54:36
what we wanted, right? And that's what has been written here also. In multi in W in in our case W, UK
54:45
and Wuv have different weights for every attention heads and as a result all the
54:50
heads have different K and V values. This solves the performance issues which we had with multiquery attention. So
54:57
here what we are doing is that with one simple trick of projecting my input
55:02
embedding into the latent space and the second trick of understanding this absorption WK needs to be absorbed with
55:09
WK these are fixed at pre-training so we don't need to cache them understanding
55:15
this absorption helps us understand that we only need to cache the CKV and if we
55:21
only cach CKV it helps us compute uh it whenever a new token comes comes in, it
55:26
helps us compute the attention scores and with the same CKV we can compute the context vector for the NE context vector
55:33
that eventually gives us the next token prediction. So we literally solve the
55:39
best of both worlds. We have we reduce the KV cache size by around 57 times and
55:45
we also maintain a good context or a good language performance which was the main issue which we had with multiquery
55:51
attention. That's the real beauty with multi head latent attention which we have learned about in today's
55:58
lecture. U so let me now recap whatever we have learned so far. To truly
56:04
understand what really happened in the multi latent attention we need to actually understand what happens when a
56:10
new token comes in. What happen when a new token comes in. So let's say a new token comes in. Right? What we first do
56:16
is that if you have understood this absorption trick, what we first do is that instead of multiplying the new
56:22
token with WQ which we did earlier, we now multiply the new token with WQ and W
56:30
transpose. So that is my absorbed query vector. Once we get this absorbed query
56:35
vector to compute the attention scores I don't need my keys vector traditionally
56:40
because because of this shuffling around I have taken this wuk which was
56:46
traditionally in my keys vector and I put it here. So to get my attention scores I don't need the traditional key
56:52
vector I just need my cache and once I have my cached latent KV matrix I
56:57
multiply my absorbed query with my latent KV cache and that directly gives me my attention score. This was the main
57:04
understanding which led to the latent attention innovation. Adding this latent space actually helps us because the way
57:10
attention scores are computed is that matrices are multiplied sequentially. So
57:15
we can absorb one matrix into another matrix and then with a simple mathematical rearrangement whenever a
57:22
new token comes in we just get the absorbed query for the new token and compute the
57:28
cached latent KV matrix. And when you multiply these two the you multiply the
57:33
absorb query with the updated KV cache. So you have to update the KV cache with the new uh with the new KV vector. You
57:41
update the latent KV cache and you multiply the Q with the updated KV cache
57:47
that gives you the attention scores. Then you get the weights and then to get the values you don't have to cache a
57:53
separate value matrix. you can use the same KV cache because if you take a look at how the context vector matrix is
58:00
calculated, you just have to uh multiply X with WD KV over
58:07
here. Um so this is my updated KV cache which you already computed and then you
58:12
just multiply it with these two. um you multiply this updated giving
58:18
cache with Wuv to get the context vector matrix and if you multiply it with WO to
58:23
get the next token. Now remember that this was already computed before. So we have already computed the updated KV
58:30
cache. So you multiply the updated KV cache with Wuv. You multiply the updated
58:35
KV cache with Wuv. Uh that gives me my values matrix. this. So this gives me my
58:42
values matrix and I multiply my attention weights. I multiply my
58:48
attention weights with my values matrix and that gives me the context vector for the new
58:53
token. That's all which I need to predict the next token anyways. So in this whole process now if I zoom out we
58:59
only need to cache this one matrix which is the KV the latent KV matrix. So the
59:05
memory requirement is only dependent on what's the dimension which I choose for the latent KV matrix. For deepsek they
59:11
chose a dimension of 576. So earlier the n into h which was the number of
59:17
attention heads into the attention dimension was 128 into 128. And we had two such caches one for keys and one for
59:24
values. But now we only have 576. So instead of 2 into 128 into 128 I only
59:29
have 576. So I get a reduction of 2 into 128 into 128 x 576 which is 57 times.
59:36
That's a huge reduction. I go from a 400 GB memory space to a 6GB memory space
59:42
for deepsek. And not just that I'm not even sharing any values across my keys and
59:48
the values right because my w my wuk and wuv are different across different
59:55
heads. So I'm not sharing any content across heads at all. Which was the main issue with multiquery attention. The
1:00:01
main issue with multiquery attention was the same content was being shared across different heads. That degraded my
1:00:06
performance. That issue is now also fully solved. This is how with the simple trick of adding that latent
1:00:12
matrix, we get the best of both worlds. We get the low cache size and we also get the good language model performance.
1:00:19
That's the beauty of this uh latent or the multi latent attention introduced by
1:00:26
deepseek. If you think about it, projecting things into latent spaces has been done in machine learning and deep
1:00:32
learning since long time. But deepseek brought it into language modeling and it's it's an incredible innovation.
1:00:38
After this lecture, we are going to see more complicated versions of the latent attention such as the decoupled rotary
1:00:44
position embedding. For that we'll first need to understand what is rotary positional embedding and how MLA is
1:00:50
modified with this decoupled rotary position embedding
1:00:55
um which they have mentioned over here but again it's a bit difficult uh to understand from their paper. So I'll
1:01:02
again break it down like I did over here. So thanks everyone. As you might be noticing the lectures are getting a
1:01:08
bit tougher but make notes along with me. In the next lecture, we are going to code the multi latent attention fully
1:01:15
from scratch. So stay tuned for that. I hoped you like this lecture. I spent a long amount of time making these notes
1:01:21
because it's very hard to find this content anywhere. It took me almost 2 months to make this lecture for multi
1:01:26
latent attention and I hoped you liked it. Thanks everyone and uh this is how deepsek changed or changed the attention
1:01:33
and rewrote the transformer. Thanks everyone and I look forward to seeing you in the next lecture.



