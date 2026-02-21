* 5:00

| Model | Context Size|
|---|---|
| gpt-4     |  8k |
| gpt-4-32k | 32k |

***

* 30:00

* to generate a new token, we only need the hidden state of the most recent token. None of the other hidden states are required.

***

* 45:00

things in memory we can store them and that reduces the number of computations
46:01
so it's good for us right what are the disadvantages well the key value cache
46:06
remember I told you that it comes with a dark side the key value cache comes with
46:12
a dark side and that is with respect to the size of the key value cache
46:17
u whenever we store something in memory memory it takes up space in the memory
46:23
so we have to pay more. So speed is important that's fine but another consideration is memory right we we are
46:30
avoiding recomp computations but the disadvantage is that we are storing something in memory caching essentially
46:36
means that we are storing something right and remember what we talked about data taking a footprint so every time we
46:43
store something it takes a footprint it's like occupying land so we have to pay the rent we have to pay the cost so
46:49
the more amount of space which is taken up by data the more cost it will incur so if you take take a closer look at the
46:56
size of the KV cache. The way the size of the KV cache is actually computed is
47:01
that let's say we have four tokens, right? The next day is every token will
47:09
have a certain number of embedding dimension that is equal to the number of
47:15
attention heads. The number of attention heads into the attention head dimension.
47:21
This is the uh dimension of every input. Now the thing is there are these many
47:26
number of inputs and the number of inputs which will be there or the number of tokens in one sentence is decided by
47:33
the context size. So if the context size is uh s in this case the context size is
47:40
four but in large models the context size is 1,000 even 10,000 1,000
47:46
etc. So n into h into s and that is for one
47:51
batch. If you have multiple batches, it's n into h into s into b. So n into h
47:56
is the dimension. S is the context size. B is the batch size. So you'll see this
48:04
n into h into s into b. So in one transformer
48:09
um the size of the cache for one transformer is this because we have to
48:14
save all of this all these number of tokens we have to save and the dimensions right. So if if there are
48:21
four dimensions 1 2 3 4 every single thing here carries some weight or rather
48:29
it occupies some space so we have to pay for it. Put it another way. Let's say we
48:35
take the same thing. The next day is right. So in this case we have used a context size of four and the dimension
48:42
of four. So n into h is four. S is equal to 1. The sorry s is equal to four
48:47
because that's the context size and the batch size is equal to 1. So for every single parameter we have to pay. So how
48:54
many parameters are there? Number of dimensions which is n into h. Number of tokens which is given by the context
49:00
size and number of batches. So here we have n into h into s into b. These many
49:06
parameters we have to save for one transformer. Now remember when we have language models there are multiple such
49:12
transformers. So we have to take that factor also which is l which is the number of layers right? Uh so l into b
49:19
into n into h into s. Now we have to have one cache for keys and one cache for values. So multiplied by
49:27
two and then here we are assuming that every single thing is a floating point
49:32
which is essentially 16 bit. So we are assuming two bytes per floating point.
49:37
So this will be further multiplied by two. So the size of the KV cache is given by L into B into N into H into S
49:44
into 2 into 2. So keep in mind here that one key variable which affects the size of the KV cache is context length. As
49:51
the context length increases the size of the KV cache increases. So we have to we have to store more memory or we have to
49:58
store more parameters. So we have to pay more. And that is the reason why we when we increase the context size remember we
50:05
saw this at the start when we increase the context size the amount of parameters which we need to store
50:11
increases because the size of the cache also increases. And that's one reason why more context size during inference
50:18
open charges us more. So this is the size of the KV cache
50:24
right and let's say if we have a 30 billion model whose number of transformer blocks is 48 the context
50:31
size is 1024 number of dimensions is 7168 and batch size is 128 this leads to
50:37
a KV cache size of 180GB that's huge if we consider deepseek R1 or their V3
50:44
model V3 base model they use number of transformer blocks as 61 if we use the
50:50
batch size equal to one during inference. Then uh the number of attention heads which they have is 128.
50:58


***



Um and the size of each or the dimension of each head is 128. And the context size is actually
51:06
100,000. So in this case if you multiply all of this the KV cache size becomes equal to 400GB. That's huge. This is
51:14
huge amount of size. And once that is stored in the memory that reduces or
51:20
that that this much amount of parameter stored in memory will reduce the other
51:26
computations also will reduce the speed of other computations also. We'll have to pay more for this much amount of
51:32
storage. So then you might be wondering how does deepse charge so less during inference. It's because they figured out
51:39
ways to deal with this. They don't use this variant of the KV cache at all. uh in fact this plot also shows that as
51:47
you so here is GPT3 small then we have GPT3 medium GPT3 large GPT3 XL so as we
51:54
go from left to right on the X-axis the GPT3 size increases and as you see as
51:59
the model size increases the number of transformer blocks increase the number of attention heads increase there you'll
52:05
see on the red on the red we have the KV cache size and the blue is the number of parameters so number of parameters of
52:12
course increase. But if you look at this red curve, it's the KV cache size that increases by a huge
52:19
amount. In fact, it increases in a quadratic or slightly accelerated manner
52:25
as the size of the model increases. That's one huge disadvantage
52:30
of the KV cache. KV cache speeds up things but it takes space. And this dark side of the KV
52:38
cache is that it takes space then we have to pay more. It reduces the speed of my other computations because my
52:44
memory is now occupied. All of this needs to be solved. And to solve this
52:50
multi head latent attention is one mechanism. And then eventually we'll
52:55
understand multi head latent attention. Once you understand this dark side of K value cache but remember this dark side
53:01
is what motivated people to later have things like multi-query attention, multi group query attention which we'll learn
53:08
about. And then ultimately deepse invented latent attention or multi latent attention to deal with this dark
53:15
side. All right, I hope now you have understood the KV cache advantages, disadvantages etc. I just want to end
53:23
this lecture by showing you a small code which I've implemented and that uh compares GPT2 inference with and without
53:31
KV cache. So let's go over here right now. Uh let me run this once more. So if
53:38
you see over here what I have done is that uh I have done something very simple. I have again used GPT2. I have
53:46
u me scroll down a bit. Yeah I have again use GPT2 which is the model which
53:51
I've used from hugging face. Then the prompt is the next day bright and I'm generating 100 new tokens. Okay. And in
53:58
one case I'm printing it with KV caching and in another case I'm printing it without KV caching. So let's run this
54:06
right now. when you run this uh so these were my previous results but I will also
54:11
show it to you u real time while running so that you have an
54:16
understanding so now actually let me run this once more to show you exactly what
54:22
is happening here the with k caching proceeds so fast that we can't even see it so now yeah see this with k caching
54:29
is already completed now it took 2 seconds and without k caching it's still running so without k caching it's still
54:36
printing ing right now as I'm speaking and it has taken 6.7697 seconds. So see the difference
54:41
here when we use cache equal to true the time taken for GPT2 to run this
54:47
inference and to produce 100 new tokens is very low. That's the advantage of KV
54:52
cache. Remember we don't need to recomputee. So the inference time becomes quick and when we use cache
54:59
equal to false the time taken increases by almost three times. So let me run this again. So you'll see with KV cache
55:05
it's done already. It took 2 seconds to print out the next 100 tokens. And without KV cache it's still running
55:11
right now and it has taken around 7 seconds. We already saw that this is the advantage of the KV cache, right? We
55:18
don't do recalculations. Um everything is done just once. We optimize or we store the
55:25
keys and values so that we don't recomputee it again and again. That's why inference time inference speeds up a
55:31
lot. That's one big advantage of KV cache. And here I have just shown these two inferences side by side. Uh so
55:38
with let me print this again. So with KV cache you'll see the printing just happens a lot more faster on the left
55:45
hand side. Again let me show it once more. Um yeah on the left hand side is with KV
55:52
cache where the inference happens much more faster and the compared to let's say right hand side. So with KV cache 52
56:00
tokens are generated in 2.5 seconds and without KV cache 52 tokens are generated in 3.6 seconds. Um so KV cache speed up
56:08
is about 1.43 times. So here I have just written a code to compare it side by side. Once I share this notebook with
56:14
you, you can also do this comparison side by side. All right. In this lecture I have
56:20
covered many things. First I have covered the main intuition of why we need a key value cache. Here we saw that
56:28
during inference during inference we are recomputing the same things again and again right we are recomputing the same
56:35
things again and again and we need to prevent that. So then we started asking the question what is my goal during
56:41
inference my goal is to predict the next token only and it turns out that to generate a new token we only need the
56:47
hidden states of or the context vector of the most recent token we don't need other context vectors. So then we
56:53
started backtracking and then we then realized that we can save or we can cache the values matrix and we can cache
57:00
the keys matrix that's all. So whenever a new token comes in I just get the query vector the key vector and the
57:06
value vector for that new token. Then I get the whole keys matrix by using the
57:11
cache. I get the whole values matrix by using the cache. Then I multiply query with keys
57:18
transpose get the attention scores. Get the attention weights. multiply attention weights of only the new token
57:24
with my whole values matrix and I get the context vector now for my new token that's all which I need to predict my
57:30
next token because this context vector will travel through the rest of this LLM architecture it will give me my logit
57:37
vector through which I'll predict my next token that's the uh that's the way we
57:43
constructed the key value cache then we saw the advantages of key value cache it gives up gives a speed up in compute
57:49
because we we don't do recomp computation. So now my compute scales linearly with number of input tokens.
57:55
Whereas if we don't use caching the compute scales quadratically and we saw this in code also we saw that when we
58:02
use caching the inference is very fast and without caching the inference is three times slower. But KV caching comes
58:10
with a dark side. That dark side is that there is a size of KV cache which scales
58:15
with respect to the number of transformer blocks, number of attention heads, the context length etc. Um and
58:22
that's why by the way GPT charges is higher for larger context length. Why? Because the higher the context length,
58:29
higher the number of transformer blocks, higher the number of attention is the size of the KV cache grows and we need
58:34
to store that much because we are caching remember. So for deep sea R1 we
58:40
saw that the size of the KV cache becomes as high as 400GB. Of course they did not use this during inference
58:46
because that would have meant that they have to pay more. So their inference price which they charge to consumers
58:52
would also increase but it's very low. How do they do that? There are now a huge number of innovations which we'll
58:58
start discussing from the next lecture onwards to understand how the KV cache memory problem is solved. The first
59:05
innovation is multiquery attention. The second innovation is grouped query attention. Deepseek did not use these
59:11
but we need to understand this before ultimately understanding multiad latent attention which is what deepseek
59:17
invented. It's an incredible technique but to understand this we need to now understand multi-query attention and
59:23
grouped query attention which we'll start seeing in the next lecture. Thank you so much everyone. I hope you enjoyed
59:29
this lecture and I took a lot of time to create this lecture particularly because I wanted to explain it from different
59:35
angles intuition theory code etc. Thanks everyone I look forward to seeing you in the next lecture.














