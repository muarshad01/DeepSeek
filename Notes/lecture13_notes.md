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

***

* 40:00


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

* 45:00



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














