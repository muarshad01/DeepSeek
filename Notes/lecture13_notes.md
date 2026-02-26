### MLA Code

* 15:00

***

* 20:00

* [B, S, D] = [1, s, 8] = [1, 1, 8]

***

* 25:00

***

* 30:00

***

* 35:00

just take this 6x4 matrix and I'll multiply it with wuv. So it's 6x4 * 4x
35:10
8. So that's my values matrix which is 6 6 by 8 and I'm again going to divide it
35:15
into two parts. So half of it will go to head number one that's 6x4 and uh the remaining half will be
35:23
head number two which is again 6x4.
35:28
In the code if we inspect where this is implemented you'll see that in the code there is this value matrix right which
35:34
is the product of um which is the product of CKV which is
35:39
my updated cache multiplied with Wuv. So that's done in this step. And here just
35:45
the values are split into two heads. And once I have the value vectors for both the heads to get the context vector for
35:53
the first head, I just multiply the attention weights for the first head multiplied by the values matrix for the
35:58
first head. So that gives me a 1x4 vector of context vector for head number one. And for the second head, I multiply
36:05
the attention weights multiplied by the values matrix for the second head. That gives me 1x4 which is the context vector
36:12
for the second head. And then I just merge these context vectors to get my final context vector which is now a
36:19
1x8. The reason it's 1x8 is because remember context vectors have the dimension equal to the output embedding
36:27
dimension which we have just chosen to be equal to the D model which is 8. So in the code if you see once we have the
36:34
attention weights what we are doing here is that we are multiplying the attention weights multiplied by the value vector
36:40
for that specific head and then we are just appending all of these context vectors together to get my final context
36:47
vector. That's it. This is the entire flow of how multi head latent attention works. Once we have the combined context
36:55
vector for that final token, then we just do multiplication with the outputs
37:01
matrix etc to get my logits vector for the next token prediction. This is how
37:06
the flow of multi head latent attention works for the next token prediction. And if you observe carefully the only thing
37:12
which we are caching here is this KV cache latent KV cache whose number of dimensions which we have decided the
37:19
number of dimensions are decided to be equal to four. So I hope through this exercise
37:24
you have understood mathematically how the dimensions work out across the different heads when we do the latent
37:31
attention mechanism. So here is the full code for this roles MLA class. Once you
37:36
have understood how every single line of this work step, every single line of this work step by step, you won't find
37:43
it that hard. Now what we are going to do is that we are going to do three types of testing, four types rather. We
37:50
are going to do a speed test. We are going to do cache test for single new inference, cache test for multiple new
37:56
inference and then we are going to visualize the WQ, W and W matrices for
38:01
all the heads. So let's get started with these tests. So in the first demo what I
38:07
want to demonstrate is the memory requirements which are saved for the KV cache when we do a latent attention. So
38:14
if you take a close look at what we are actually caching we are just caching the latent KV cache which is the product of
38:21
my input multiplied by WDKV. So my input is projected into a latent dimension
38:26
space. So there are three dimensions which govern this size. The batch size, the context length and my latent
38:33
dimension. Right? If we have normal multi head latent attention then also we have the batch size then we have instead
38:41
of the latent dimension we have the model dimension u which is the number of attention heads
38:47
multiplied by the head dimension which is 512 in this case. Um and then we
38:53
have a two here which is basically we have one cache for keys and we have one cache for values. One more two here is
39:01
because we are assuming a floating point 16 which consists of two bytes. So this
39:07
one two is present here also. But when we consider the multi latent attention
39:13
we just have one um latent matrix whose size is now 256 because my KV latent
39:19
dimension is 256. So out of all of these values here I want to I want you to only focus on
39:25
two values. First we look at when we look at multi head multi head attention we use the D model which is equal to 512
39:33
but in latent attention we use the KV latent dimension which is 256 in this
39:39
particular demonstration which I'm showing right now. So here already we have reduction of two times then we have
39:45
another reduction of two times here because in the standard multiate attention we had one cache for keys and
39:51
one cache for values. That's why we had two caches. But here there will only be one cache. So there will be a total
39:58
reduction of four times. Um and if you print this out, you'll see that the
40:04
standard let's say takes 80 kilobytes or 80 KB and the latent KV cache. So
40:11
actually I think I'll have to run the first code also. So I've run the first code right now where the roles MLA class
40:17




***

is uh defined and then I've run my uh second code over here. So the main thing which
40:23
I want to show you is the KV cache reduction which we get of about four times here. So intuitively um even if
40:30
you forget the mathematics later just remember that the latent when we use the latent attention the KV cache size
40:36
reduction is obtained because of first the latent head dimension we can choose it to be lower than the output dimension
40:43
of the model and second is this factor of two which is now not there. That's
40:48
our first testing. The second testing which I want to show you is how the cache gets accumulated whenever new
40:54
tokens are inferred. So if you see the cache size is governed by B comm S,
41:00
latent dimension. Right? So this S is essentially number of tokens which are inferred until that point. So as the
41:07
number of inferred tokens increases my cache size should increase. Right? So what I'm doing here is that I'm just
41:13
showing you an initial input which has a cache of 1a 5a 4. And
41:19
when a new token is added, the cache shape essentially becomes 1, 6, 4, which
41:24
means that the cache size actually increases when a new token is inferred and it's appended to the
41:30
cache. So initially we had five tokens and then we had add we have now added one token. After this point, I have
41:36
written a custom loop where we can test multiple new inferences. Right? So if initially we had 50 tokens in the cache,
41:42
when you run this, you'll see that uh then the 51st token is inferred. the cache size increases to 51. The 52nd
41:50
token is inferred. The cash size increases to 52. Then the 53rd token is inferred. Then the 54th token is
41:55
inferred, etc. That's how the cache size actually goes on increasing during the inference. Now remember what we started
42:03
out with the multi head latent attention gives us the best of both worlds, right? The multi head latent attention actually
42:11
um reduces the size of the KV cache which we saw in the code. But another
42:17
advantage is that it does the multi head latent attention does not have the same values or content across the heads. It
42:23
does not share any content across heads. That's why the performance is retained.
42:28
What I want to now demonstrate is that if you look at the schematic of latent attention, I want to show you that WUK
42:36
Wuv these matrices don't share any content across the different heads. That's why the keys
42:43
and the values which I obtain are different for different threads and that makes the latent attention performance
42:50
much better than a multi-query attention or a grouped query attention. So what we can do now is that we can visualize the
42:57
WQ W and WV matrices for different heads. So I'll just run this code. So
43:04
what I've done here is that I have taken a simple model with with has which has the output dimension equal to 8 and the
43:10
two heads which is the same case which we considered on the whiteboard the output dimension equal to 8 and we have
43:17
two attention heads and what I have visualized here is that I've visualized first you can look at WUK I visualized
43:25
the first head and the second head right what are the dimensions of WUK the
43:30
dimensions of WUK are essentially uh the latent dimension multiplied by
43:36
the output dimension right and the output dimension will be split across the two heads. So the first head will
43:42
have dimension 4x4. The second head will have dimension 4x4. That's why you'll
43:47
see both heads here have the dimension of 4x4. Head 0 has dimension 4x4. Head 1 has dimension 4x4. Similarly for wuv
43:56
head 0 has dimension 4x4 and head 1 has dimension 4x4. What I want to show here
44:02
is that the head 0 content and head 1 content are completely different from each other for the keys as well as for
44:09
the values. This is the main difference between latent attention and grouped query attention or multi-query
44:15
attention. No content is shared between heads at all. And if you here I've taken
44:21
a much more complex example where the model output dimension is 32 and the number of heads is equal to 8. KV latent
44:28
dimension is also equal to 8. So if you look at Wuk now it will have a size of
44:34
8x4. Why 8x4? because the latent dimension is because the latent dimension is 8 and the model dimension
44:42
is 32. So each head will share four dimensions of the model and latent dimension will be eight. So each matrix
44:48
is 8x4. So here I want you to see that for head zero, head 1, head 2, head 3. I
44:54
have shown four heads for reference. Actually there are eight heads but you'll see that the values are
45:00
completely different across heads for the WUK and for Wuv. No two heat maps
45:05
are the same at all. Which means all the heads share or all the heads don't share anything for the keys as well as the
45:12
values. And that's the major reason why u latent attention maintains performance
45:19
which was completely lost with multi-query attention and group query attention. That's why latent attention
45:26
uh latent attention reduces the KV cache size but at the same time it maintains a very good language performance. Hence it
45:33
achieves the best of both worlds. Okay. So I hope through this lecture I have
45:38
showed you two things. First I wanted to show you mathematically how the when a
45:44
new token comes in what actually happens in terms of matrices. I truly believe that to completely understand the
45:50






***

concept. We don't just need to see the code. We need to write the matrices down on a piece of paper and work out
45:56
everything by hand from scratch. So the whole goal of today's lecture was to go
46:03
right down to the level of matrix multiplication and understand latent attention and parallelly write a code um
46:10
for the latent attention block. And I wanted to show you that it's not as hard if you understand what's going on
46:16
beneath the scenes if you can write it down on a piece of paper. And then I also wanted to show you main
46:22
demonstrations with respect to the fact that the latent attention does achieve a reduction in the KV cache memory and at
46:30
the same time it maintains language performance because it does not share any content across the different heads
46:35
and I hope I have been able to get these points across. Thanks a lot everyone. Please make notes as I'm speaking
46:42
because lectures are getting progressively more difficult. So to follow along it is extremely critical that you make detailed notes. In the
46:49
next lectures we'll be looking at rotary positional encoding. Then we'll modify the multi head latent attention with
46:55
rotary positional encoding. And then after that we'll look at mixture of experts multi-token prediction etc. So
47:02
lots of cool things are going to come. Thanks a lot everyone and I look forward to seeing you in the next lecture.












