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

* 15:00

* To get the logits vector for "bright," we only need the context vector for "bright".

* __4__: What if we store/cache the keys and valus matrices during inference?
* __5__: We need to cache Kay and Value matrices. This is called a K-V cache. We don't need to store Queries (Q) matrix.
* __6__: K-V cache advantages.
  * Computation Cost = O(number of tokens)

***

* 20:00

* __7__: K-V cache disadvantages.
  * Size of K-V cache

* __8__: Solving the K-V cache memory problem

* 30:00

* __9__: Can we get best of both worlds?
  * Low cache size
  * Good language model performance

***

* 35:00

* __10__: What if we don't have to cache the Keys & Values seperately?
  * What if we cache only one matrix.
  * What if this matrix has less dimention than "n X h"
 
 $$2 \times n \times h \to 1 \times n_l \times h$$

* To get this matrix, we start by projecting the input embedding matrix into __latent space!__

$$X: (4,8) \times W_{DKV} (8,4) \to \text{Latent Matrix}: C_{KV}(4,4)$$


***

* 40:00


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


***


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



***


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

















