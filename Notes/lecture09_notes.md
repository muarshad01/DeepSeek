* 5:00

| Model | Context Size|
|---|---|
| gpt-4     |  8k |
| gpt-4-32k | 32k |


***

* 10:00

***

* 15:00

***

* 20:00


new input sequence is the next day
20:17
bright is that again goes to my through my entire
20:25
pipeline and I have my next token which is let's say
20:31
and and you see after every inference the token is appended to the next
20:38
input. So already you must be seeing here right? First this input sequence goes through the entire pipeline. Now
20:44
this input sequence goes through the entire pipeline. This input sequence then again goes through the entire
20:49
pipeline. It seems that a huge number of computations need to be performed during the inference. And not just that
20:56
intuitively it seems that we might be performing the same calculations again and again. Right? Which means the next
21:02
day I already passed these three inputs through the entire architecture. Now do
21:07
I have to again pass these three inputs through the entire architecture? Then do I have to again pass these three inputs
21:13
through the entire architecture? It seems that we are just performing the same computations again and again and
21:19
again. Right? We are passing the same tokens in the input sequence through the entire architecture once a new token is
21:25
generated. So just understand or intuitively imagine you have constructed the LLM and
21:33
you are getting the inference but you think about the fact that actually a lot of in computations need to be performed
21:40
during inference and maybe we are repeating the computations and what is the problem
21:45
with repeating the number of computations. The main problem with repeating the number of computations is
21:50
that for every piece of data stored in the memory we pay a price. Think about data as houses, right? Imagine you have
21:58
a huge land and there is a house. For every land area occupied by the house,
22:04
you have to pay a price, right? Similarly, think of the memory as that area which we have which is very
22:10
precious to us. For every piece of that memory occupied with data, we have to
22:16
pay a price. Okay? And uh the more the memory is occupied,
22:22
the higher the price we have to pay during inference. The thing is the more number of computations you perform, the
22:28
more number of repeated computations you perform, the more amount of data you need to store in the memory, that leads
22:34
to more amount of computations that leads to more price which you have to pay during the inference. That is one
22:41
reason intuitively why a higher context size is priced higher. We'll come to why context size matters in a moment. But
22:47
just remember that more context size means more memory uh and every piece of
22:53
data stored in the memory we pay a price. So now until now what we have seen is that many computations seem to
23:00
be repeated during the inference. That is number one. Second insight which we have is that actually during the
23:06
inference only the context vector for the last token matters. So maybe we can
23:11
do something more efficient. So then the question is can we do something to reduce the memory storage
23:18
during inference and once you start asking this
23:25
question that once you intuitively realize that okay I seem to be doing so many repeated computations during the
23:31
inference can I do something to reduce the memory storage and this is where the
23:36
key value cache or this is where the KV cache actually comes into the picture we'll now See
23:43
mathematically so we'll prove that we are actually repeating the computations. Currently I just intuitively showed you
23:50
that we might be repeating some calculations. Right now we are actually going to take an hands-on example and
23:55
I'm going to show to you that during inference we are actually repeating many computations. Then we will see what to
24:03
do to avoid those repeat repetations. And to implement that logic of avoiding
24:09
repetitions, we are going to use this intuition which we have seen that we only need the context vector for the
24:15
last token. So I hope you are excited for the next part of this class where we are going to dive a bit into visual
24:24
mathematics. All right. Now I want to prove to all of you that we are repeating calculations or repeating
24:30
computations during inference and many things can be optimized further. Let me prove that to you. So first we have to
24:38
look at the self attention mechanism and we have seen this a lot in the previous classes. If you are not aware of this
24:44
mechanism, I encourage you to go through the previous classes. U so here's what
24:49
happens during the attention mechanism. Right? Let's say the input is the next day is okay. So imagine now that when we
24:57
are looking at the attention mechanism, we are focusing on this block right now. Okay. So here's what happens. The input
25:05
is the next day is that's the input sequence and we are considering an 8dimensional input sequence and we have
25:12
four tokens. That's why the input embedding matrix is 4x8. We multiply it with the trainable query weight matrix,
25:19
trainable key weight matrix and trainable value weight matrix. That gives us the queries matrix, the keys
25:24
matrix and the values matrix. Okay, all of these are 4x4. Then we multiply the
25:30
queries with the keys transpose and that gives us the attention scores. Every row of the attention score
25:37
corresponds to one query which let's say is this and how it attends to the other tokens in the input sequence. That's why
25:44
the attention scores is 4x4. The size of attention scores will always be number of tokens. Number of tokens, number of
25:53
tokens. Since we have four tokens here, the next day is it's a 4x4 matrix. Then
26:00
once we have these attention scores, we scale it by square root of the keys dimension. We apply soft max and apply
26:07
causality so that all the elements above the diagonal are zero. These are our attention weights and then we are going
26:13
to multiply the values vector or we are going to multiply the attention weights with the values matrix and that gives us
26:20
the context vector matrix. So once we come out of this attention block here we
26:25
have this context vector matrix right. So then we have context vectors for every or sorry the context vector for
26:33
the next day and is. So these are the context vectors they are much more
26:39
richer than the input embedding vectors because each context vector captures information of the neighbors. Okay
26:46
that's how the self attention mechanism proceeds. Now we are doing inference.
26:52
Always keep this in mind. This entire lecture is in the domain of inference. We are not doing pre-training. So now
26:58
let's say we have to predict the next token. Uh we have next day is and during
27:05
this entire pipeline once we have this context matrix it goes through all of these rest of the layers. We get the
27:11
logits matrix and we predict the next token. So the the next day is goes through this entire transformer block
27:18
during info during inference. We get out we get the logits matrix and the next token is predicted. Let's say the next
27:24
token is bright. Now uh imagine or remember what I told you
27:30
the bright which is the next token will be appended to the previous input. So now during the next inference stage my
27:37
new input matrix becomes the next day is bright. Correct? So now my input matrix
27:43
is 5 by 8 because there are five tokens now. And now this input matrix will
27:48
again go through the entire attention mechanism. We'll multiply it with the queries keys values. We get the queries
27:54
keys values. Now those are 5x4 5x4 5x4. The attention scores matrix is 5x 5. The
28:02
attention weights is 5x 5. And the context matrix is now 5x4 because the
28:08
first is for every or sorry the first row is for the next day is and the last
28:14
row is for bright. Okay. So
28:20
now take a look at these computations and take a look at the earlier
28:25
computations which we did. Do you notice something
28:31
repeating? So in the earlier computation there were four tokens right the next day is right and in the next computation
28:40
there were five tokens the next day is bright wq wk and we are the same
28:46
matrices in in this step and uh in this step. So what is
28:52
repeating when we computed in the previous inference step and when we computed the attention scores in the
28:58
next inference step? What is exactly repeating? You can pause this video for a while
29:05
here. Okay. So now let's say if you take a very closer look at this new uh
29:12
context matrix calculation, you will see that everything which I've shown in this black box is repeated.
29:21
So these three which I have marked in the black box let's say these are 4x4 right the first black box is 4x4 the
29:28
second black box is 4x4 and the third black box is 4x4 we already computed
29:33
these queries keys and values 4x4 matrices in the previous inference
29:39
step is the first repetition which you should be aware of then let's look at the attention scores in the attention
29:46
scores currently we have a 5x5 Okay, actually let me remove this. We
29:53
have this 5x5, correct? But now I'm marking one 4x4. This 4x4 which I have marked over
30:01
here, we have already computed this 4x4 attention score in the previous
30:06
computation. We have already computed this. Similarly when we go to the attention
30:12
weights now Similarly when we go to the
30:18
attention weights now let me mark this we have already computed these
30:23
4x4 in the previous attention weights calculation because previous attention weights was also 4x4. So you see the
30:30
problem which we are or the repetitions which we are doing here. We are again repeatedly
30:37
calculating the same things again and again. We are recalculating the queries keys and values. We are recalculating
30:43
the previous attention scores. We are recalculating the previous attention weights. We are also recalculating the
30:49
these four values of the context vector matrix which was already computed before. So why are we doing all of these
30:56
rec computations? In fact, keep in mind that our whole goal of this task is to
31:02
predict the next token. Correct? Our whole goal of this task is to predict the next token. And from our learning
31:09
before to predict the next token after this, what do we need to predict the
31:14
next token? All we need is that to predict the next token, we only need the
31:20
context vector for the last token. Which means that after doing all of these repetitions, the only thing which we are
31:26
going to use is this context vector for bright because that's my last token
31:32
currently. This context vector for bright is the only thing which we need to predict the next token. So why are we
31:38
computing all of these things unnecessarily. So now let us formalize
31:44
this and try to quantify what are we computing again and again and what can we do to solve it. So think about this
31:51
right? It looks like we are unnecessarily doing a lot of repeated calculations. Can we optimize this? And
31:57
already you can think that one way to optimize this is that these query or these keys and values matrices which
32:03
I've computed previously, why do I need to compute again? Can I just store the
32:10
keys and values matrices from my previous calculation?
32:15
That's where the concept of caching comes into the picture. Caching is basically just storing the previously
32:21
computed values so that you can use them in the next iteration. So then you might be thinking
32:27
let me store the queries also. Let me store queries, keys and values. Let me store all of that. Now let's come to
32:34
what exactly we need to store. Okay. Now we have understood that repeated
32:39
calculations are happening and we need to store something in memory to make our computations more efficient as the next
32:46
stages of inference proceed ahead. Now we are going to see step by step what exactly we should store. Okay. The input
32:53
is the next day's bright and our goal is to predict the next token. So the next day is bright will travel through this
32:59
entire architecture and we have to predict the next token. Remember what I told you earlier to get the next token
33:06
prediction. When when we reach this final layer when we reach this final layer we'll have a logits matrix. We'll
33:12
have a logits matrix for the next day is bright. This is the logits matrix. When
33:19
we reach the end of this computation, so when we reach at this part, we'll have the logits matrix. And the size for each
33:26
will be my vocabulary size. All of these. Now, out of all of
33:31
these vectors, I don't care about these first four at all. I don't care about
33:36
these because I want to predict the next token. I only care about the logits vector for bright. Then I'm going to
33:42
look at that index with the highest probability and predict the next token. So I only care about the logit vector
33:49
for brite. As a result, I only care about the context vector for bride which
33:56
is computed at this stage. I only care about the context vector for brite. After this stage all the other tokens do
34:03
not matter to me at all. So after this point once I get the logits vector for
34:08
bright or sorry once I get the context vector for bride my context vector can go through all of these layers and only
34:15
that context vector influences my logits vector so I only need that I don't need other tokens after that point so after
34:22
this point which is marked in orange right now the earlier tokens are not
34:27
needed to generate a new token we only need the uh hidden state or the context
34:33
vector of the most recent token none of the other context vectors are needed. Keep that in mind. So when you look at
34:40
this context vector matrix, when you look at this context vector matrix you had obtained earlier, we don't need the
34:46
entire context vector matrix. We only need the context vector which is corresponding to brite. That's the most
34:53
important realization. Once you realize that you only need the context vector for brite, let's backtrack now. Okay. So
35:01
we only need this. That's what we have understood up till now. We have got the whole context vector. But I don't need
35:07
these other context vectors at all because they don't influence this final logits vector. So I only need the
35:13
context vector for bride. Now let us backtrack and check what we actually really need to get this context vector
35:20
for bride. What do I need? How will this context vector for bride be calculated? I need the attention weights only for
35:27
bride and I will multiply it with the entire values matrix. that will get me
35:32
the context vector for bride. So I need the attention weights for bride and I need the values matrix. How do I get the
35:39
attention weights for brides? To get the attention weights for brightes, I need the attention scores for bright, which
35:46
means how bright how bright relates to all the other tokens, right? The next day is
35:54
bright. I need to get these attention scores. These are these attention scores
35:59
1, five. So I need five values 1x five. How do I get these attention scores? I
36:05
get these attention scores because I need only the query vector for bright multiplied with the keys
36:11
transpose. That's all. So I will need the So here is what
36:17
I will actually need. I will need my values matrix. That is what I will need.
36:23
And I will need my attention weights. To get the attention weights I need attention scores. To get attention scores, I need query vector for bright
36:30
multiplied by the keys transpose. Now, how do I get the values vector? How do I get the uh values or
36:38
how do I get the values matrix? Let's say this is my whole values matrix. So, let's start from the top. To
36:45
get the context vector for bright, I need attention weights for bright multiplied by values matrix. What I have
36:51
marked in my black box is the cache value matrix which is the value matrix
36:56
coming from the previous iteration. So the top four rows I'll get from my cache. My bottom row which is the value
37:02
vector corresponding to bride. I just take the input for bride multiplied with
37:08
the trainable weight matrix for bride. And this is how I get the value vector
37:13
for brite. This is the only new computation which I have to do. For all these other value vectors, I can anyway
37:19
cache them from my previous iterations. Then to get the attention
37:26
scores, we need the query vector for bright multiplied by the keys transpose. To get the query vector for brite, I
37:32
just take the input for brite multiplied with the trainable query matrix, trainable query weight matrix. And if
37:39
you look at the keys transpose, same two values, same as the values, the first four rows of this can be cached. I don't
37:46
need to recomp compute this again. But only the last row which is the keys corresponding to brite will be the input
37:53
corresponding to brite multiplied by the trainable key matrix. That's all. So if I zoom out a
38:00
bit these three boxes number one, number two and number three are the only three
38:06
new computations I need to do for every inference. So take a look closely at
38:12
these three boxes. What are these boxes? These boxes are just the input vector for brite multiplied by the trainable
38:19
query, the trainable key and the trainable value matrices. And these trainable matrices, this W, Q, WK, and W
38:27
are already fixed because WK, W, Q and W are fixed during pre-training. We don't
38:32
need to compute them again. So they are already fixed. I don't even need to catch them. They are fixed values. They
38:39
are fixed during pre-training. So I get my input token. So once a new token
38:44
comes in, here's what I have to do. Once a new token comes in, I find the query vector corresponding to the new token
38:50
first by multiplying the input embedding for that token multiplied by WQ. Then I
38:56
get the query vector for the new token. Okay? Then what I do is that I have already cached my previous keys. Then I
39:04
compute the new key vector for the new token. I compute the key vector for the new token by multiplying the input
39:10
embedding with W K. This is the key vector for the new token. I do not compute the key vectors for the previous
39:16
tokens because they are cached. Then I augment my new key vector
39:22
with my previous cache to get the whole keys matrix. Then I multiply the query vector with the keys transpose. I get my
39:29
attention scores. I scale them. Apply soft max and causality to get my attention weights. Once I have my
39:36
attention mates weights, I calculate the values vector for the only the new token
39:42
by multiplying the input embedding with we get the value vector for the new token. And then to get the values
39:48
matrix, I just append the new values vector with the cached values. So I don't compute this cache again. And then
39:56
I multiply the attention weights multiplied by this value vector and I get only the context vector for bride.
40:02
That's it. I only get the context vector for bride. I don't care about the other context vectors at all. So again, if I
40:10
zoom out, pay careful attention to how many caches I need. I only need to cache
40:15
the keys and I only need to cache the values. I don't need to cache the
40:20
queries at all because I just need the new query vector for my uh for my new
40:26
token. Since we only need to cache keys and values, this is called as key value
40:34
cache. This is called as key value cache. And sometimes it is also referred
40:40
to as the KV cache. So once we cache the keys and the values, once we cache the
40:46
keys and the values, every time a new token comes in, we don't need to recomputee those previous keys and
40:52
values. So again, let me repeat what happens when a new token comes in. When a new token comes in, I first multiply
40:59
the input embedding with WQ, W K and WV. Only three computations need to be done.
41:04
Then I get the query vector, the key vector and the value vector for the new token. Based on the query vector, I
41:12
multiply the query with the keys matrix. So to get the keys matrix, I use the keys cache and append my new keys
41:18
vector. So then the query vector multiplied by the keys transpose gives me attention scores.
41:25
uh then I get the attention weights and then I get the I use the cache value
41:30
matrix and append the new value vector for the new token. That's how I get my
41:36
new values matrix. Then I multiply my attention weights with the new values
41:41
matrix and that's how my I get my context vector for the new token. That's it. Then we get the context vector for
41:48
bright. And then this context vector at this stage then it passes through the rest of the architecture. When it comes
41:54
out we get the logits vector. We get the logits vector only for
41:59
bright. Then we look at this logits vector and we find that index with the
42:05
highest probability and that gives us the next token. This is what is meant by key
42:11
value cache. we just store the previous keys and values. But to understand key value
42:17
cache, it is very important to understand this intuition that to generate a new token, we only need the
42:24
context vector of the most recent token. In this case, it was bright. And this
42:29
insight actually helped us that helped us to know what to cache. So once we got
42:34
this insight that I only need this new context vector, then you see how we backtracked. Then we know what all to
42:40
cache, right? then it becomes very easy. We only need to cache the keys and the values. We don't need to cache queries.
42:46
Uh that's all and only three new computations need to be done every time a new token comes in. The token
42:52
embedding multiplied by WQ, WK and WV. That's all. Then the keys are appended
42:57
to the cache. The values are appended to the cache and we get the context vector for the new token. That saves a huge amount of
43:04
computations during inference. We are we don't need to compute every single thing again and again like we saw over here
43:11
these black boxes were recomputed again and again right all right so until now what we
43:19
have learned is that we need to cache the keys and the values matrices and this is called as the key value cache.
43:26
We don't need to store or cache the queries matrix at all. Now what's the use of the key value cache? The main use
43:32
of the key value cache is that as the number of input tokens increases, the amount of compute time increases
43:39
linearly which means that we can speed up the computations by a huge factor.
43:45
Why do we speed up the computations? Because we are not doing repeating calculations again and again. Earlier if
43:51
you notice what was happening what was happening here is that this the next day
43:56
we get bright. Bright is appended then the next day is used again for the next calculation. then it's used again for
44:02
the next calculation. These repeated calculations lead to quadratic computations or quadratic
44:08
complexity. What quadratic complexity actually means is that as the number of tokens
44:14
increases, as the number of tokens increases, if we don't do caching, the
44:20
amount of computations which we need to do increase in a quadratic manner. But once we do caching. So once
44:28
we do caching what actually happens is that if you see the same example this
44:34
once it's used is cached. It's not computed again. It's cached. Then once this is used it's cached. These keys and
44:41
values are cached. So we don't need to compute this again and again. As we saw the keys and values vector are cached.
44:47
So we don't need to compute these black boxes again and again. This caching helps us because it ultimately leads to
44:53
a linear compute time. This caching leads to a linear compute
44:59
time instead of a quadratic compute time. So as the input tokens increase the compute time increases linearly not
45:06
quadratically. So once we use the k value cache computation speed up uh
45:12
because we don't recomputee the same thing again and again and again. Uh that's the whole advantage of
45:21
using a key value cache that we can just store the variables in memory and then
45:26
we don't need to reuse or we don't need to recomputee the same thing again and again. Remember when we recmp compute
45:33
the same thing the number of computations increase uh then the cost also increases as we saw earlier every
45:39
single compute instance takes cost. So the moment we reduce the number of
45:44
computations we reduce the cost and that is the key advantage of key value cache.
45:50
So it seems that everything is amazing right this key value cache we can cache
45:56
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






