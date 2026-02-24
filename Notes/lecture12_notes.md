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

$$(W_Q,W_K,W_V) \to (W_{Q},W_{UK},W_{UV})$$
* Note: W_Q remains the same but W_K and W_V and projects to W_{UK} and W_{UV}

#### How does adding this latent matrix help?
$$
\begin{aligned}
Q      &= X \times W_Q \\
C_{KV} &= X \times W_{DKV} \\
K      &= C_{KV} \times W_{UK} = X \times W_{DKV} \times W_{UK}\\
C      &= C_{KV} \times W_{UV} = X \times W_{DKV} \times W_{UV}
\end{aligned}
$$

#### The Absorption Trick

$$
\begin{aligned}
\text{Attention ~Score} &= Q \times K^{T} \\
                        &= XW_Q \times (W_{UK}^{T} \times W_{DKV}^{T} \times X^{T} )\\
                        &=X(W_QW_{UK}^{T}) (XW_{DKV})^{T}
\end{aligned}
$$

* $$X(W_QW_{UK}^{T})$$: Absorted Query - Fixed at training time (only compute once).
* $$(XW_{DKV})^{T}$$: This needs to be cached.


$$
\begin{aligned}
\text{Context Vector Matrix} &=  \text{Attention Weights} \times V \\
                             &= (QK^{T})(XW_{DKV}W_{UV}) \\
                             &= (QK^{T})(XW_{DKV}W_{UV})W_{0}: \text{Logits Matix}\\
                             &= (QK^{T})(XW_{DKV})(W_{UV}W_{0})\\
                             &= (\text{Attention Scores})(\text{Cached})(\text{Fixed at traing - Only commputed Once})\\
\end{aligned}
$$




***

* 45:00

#### Example

#### Letent KC-cache

* $$X(4,8) \times W_{DKV}(8,4) \to KVcache (4,4)$$

* __15__: So, what happens when a new token comes in?
* First, we compute the queries project into latent space.

$$
\begin{aligned}
Q &= X_{bright}(W_Q.W_{UK}^{T})\\
&=X_{bright}(1,8)(8,4)(4,4) \to \text{Absorbed Query vector for bright}(1,4)\\
\end{aligned}
$$

* __2__: Compute KV Vector
$$
\begin{aligned}
Q &= X_{bright}.W_{DKV}\\
&=X_{bright}(1,8).(8,4) \to (1,4) \to \text{Append to latent KV cache}\\
\end{aligned}
$$

***

* 50



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





***


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









***


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

***






