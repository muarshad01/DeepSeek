#### The Absorption Trick

$$
\begin{aligned}
\text{Attention ~Score} &= Q \times K^{T} \\
                        &= XW_Q \times (W_{UK} \times C_{KV})^{T}\\
                        &= XW_Q \times (W_{UK}^{T} \times W_{DKV}^{T} \times X^{T} )\\
                        &=\underbrace{X(W_QW_{UK}^{T})}_{Fixed ~at ~training  ~time}~\underbrace{(XW_{DKV})^{T}}_{This ~needs ~to ~be ~cached.}
\end{aligned}
$$

* $$X(W_QW_{UK}^{T})$$: Absorted Query. Fixed at training time (only compute once).
* $$(XW_{DKV})^{T}$$: This needs to be cached.

***

* 10:00

$$
\begin{aligned}
\text{Attention ~Score} &= Q \times K^{T} \\
                        &= R_{pos}(XW_Q) \times R_{pos}(W_{UK} \times C_{KV})^{T}\\
                        &= R_{pos}(XW_Q) \times R_{pos}(\underbrace{W_{UK}^{T} \times (XW_{DKV})^{T}}_{We ~need ~to  ~recompute ~keys ~for ~all ~tokens}) \\
\end{aligned}
$$

* This will significantly hinder inference efficiency.
* Note: $$W_QW_{UK}^{T}$$ can't be directly absored.

***

* 15:00

***

* 20:00

#### Decoupled RoPE

$$
\begin{aligned}
\text{Attention ~Score} &= Q \times K^T \\
                        &= [Q_C : Q_R] [K_C:K_R]^{T}\\
                        &= \underbrace{Q_CK_C^T}_{Ratain ~old ~magic ~MLA} + Q_RK_R^T
\end{aligned}
$$

* $$W_Q \to W_{DQ} \to W_{UQ}$$

***

* 25:00


***

* 30:00

|||
|---|---|
|$$d$$||
|$$d_{C}$$||
|$$d_{C}^{/}$$||
|$$d_h^R$$||
|$$n_h$$||
|$$d_h$$||

***

* 40:00

$$Q_C = X(W_{DQ}W_{UQ}W_{UK}^T)$$

***

* 50:00

first thing we have to do is compute CQ which is down project the input embedding by multiplying with WDQ.
50:34
That's exactly what we are going to do. We are going to uh take my input embedding vector for bride which is 1a
50:40
8. We are going to multiply it with wtq which is 8x4. Four is DC uh DCash which
50:47
is the latent down projection which we also saw over here. Latent down projection in the
50:54
query space. So my CQ will be 1x4 and I will multiply it with WQR which is 4x8.
51:00
That will give me my 1x8 vector the query vector for the new token. and I'm
51:06
going to apply rotary positional encoding to it which gives me my QR. We don't need to cache this at all because
51:12
that's the query vector for the new token and I have my uh KR updated KR
51:17
cache which is 5A 8. So to get the attention scores for bright in this second path we just multiply the query
51:24
vector QR 1 by 8 multiplied by updated KR cache transpose. So multiplied by 5
51:30
comma 8 transpose. So this will be 1a 5 that will be the attention scores for
51:36
bride computed through the second pathway. So through the first pathway the attention scores for bride is 1a 5.
51:42
Through the second pathway also the attention scores for bright is 1a 5. Then we are what we are going to do is
51:48
we are going to add these two vectors together and that is going to give me my resultant attention scores 1a 5. So if
51:54
you zoom out and take a look at this entire calculation, there are only two things which we need to cache. For the
52:00
attention scores path one, we only need to cache my CKV which has the dimension
52:06
equal to DC. And in my attention space number
52:11
two, I only actually need to cach KR. And since we are actually only repeating
52:17
in the KR, we are repeating across heads. I only need to cach DHR. I don't need to cache for multiple heads. I just
52:24
need to cache for one head DHR. So the total dimensions which I'm
52:30
actually caching is DC plus DHR. I believe the reason they share the
52:37
values in the KR across multiple heads is so that we can reduce the computational memory because when we
52:43
ultimately see the KV cache size it will be dependent on DC and DHR. So actually if different values were there across
52:49
heads the total dimensions which we will need to cache would have become DC plus NH into DHR. But now since we are
52:57
sharing it across heads it will just become DC plus DHR. So once we only cach the CKV and
53:05
the KR we get the resultant attention scores that's 1A 5 and we already have
53:10
the updated KV cache 5A 4. We multiply it with Wuv 4A 8. That gives us the VC
53:17
which is the values. And then we multiply the resultant attention scores with the values. And that gives me my
53:23
context vector for bright which is 1a 8. And for the next token prediction I only
53:29
need the context vector for my last token. So this ultimately gives me my next token prediction. Right? So right
53:36
now in this section what we saw is what happens when a new token comes in. When a new token comes in we have two
53:41
pathways. First pathway I use my absorption trick. I get the attention scores and I only need to cach CKV. The
53:50
second pathway I get the Q I get the KR. I cash it. I get the QR and I multiply
53:57
QR multiplied by KR transpose and I get my attention scores for bride through the second pathway. Then what I'm going
54:03
to do is that I'm going to add these two attention scores and that gives me my resulting attention scores for bride.
54:10
That's a 1a 5 vector. Then these are the resultant attention
54:15
scores for bright. Then we already have the updated KV cache, right? And we have already found and we have already have
54:23
the Wuv. So we multiply the KV cache with W UV to get my
54:29
updated values ve values matrix which is 5A 8 and the attention scores will be
54:34
multiplied with the values matrix and that gives me a 1x8 vector which is the context vector for
54:40
bright that's it and this context vector is then used for the next token prediction task. This is the whole
54:47
pipeline uh which is there for new inferences of
54:52
tokens. And the beauty of this pipeline is that we are definitely reducing the cache size as we'll come to later. But
54:59
throughout this entire pipeline, we are not sharing information across heads. Right? In the left hand side, we are not
55:05
sharing any information across heads. In the right hand side, the only place where where we are sharing information
55:10
is here. uh the only place we are sharing information is KR. So that's why memory
55:18
is reduced through the KV cache size but performance is also maintained because we are not sharing too much information
55:24
across heads. Now what we can do is let's come to the last part where we actually
55:29
compare the KV cache size across all the variants of the attention mechanism
55:35
which we have seen so far. So here's the KV cache memory size comparison. Remember the whole thing started with
55:42
reducing the KV cache memory size but at the same time we want to maintain the performance right. So the original KV
55:48
cache size in multi head attention was 2 * 2 multiplied by LB SNH. What are these different things?
55:55
This h is my uh attention head dimension. N is the number of attention
56:00
heads. S is the number of um S is the context length. B is the
56:07
batch size. L is the number of transformer layers. I'm assuming a 16
56:13
bit representation for each parameter. So we have two bytes over here. And the reason this one more two is there is
56:20
because we have a k cache and we have a vcache. So k and v. So that's why for the original multi attention it's this
56:26
much memory size. What MQA and GQA tried to do is that they tried to share values
56:32
across heads. So what that did is that instead of having this n in multi
56:38
multiquery attention we get rid of the number of attention heads completely. So that saves my memory size a lot but that
56:44
degrades my performance by a huge amount because uh my value my different my
56:50
content is shared across different heads. So the diversity is not there in the attention scores and my models just
56:57
don't perform well on various evaluation benchmarks. Even GQA if you see they
57:02
they are better than MQA because instead of sharing values or instead of replicating values across all heads they
57:08
divide it into groups. So each group has the same or one group then second group
57:14
third group fourth group etc. So each group has the same value but different groups have different value among one
57:19
group all attention and share the same value. It's a bit better than MQA but still the performance is not quite good.
57:27
So if you see actually uh in appendix D1 I think in the deepsek
57:34
paper they have a comparison between um MQA, GQA and multi head attention the
57:41
deepseek paper and here you'll see that MQA and GQA perform much worse on all
57:47
these MMLU benchmarks CMMLU benchmark they perform much worse than the multi head attention. So although the KV cache
57:54
size is reduced for MQA and GQA their performance is not good at all. So ideally we wanted something which
58:01
maintains the performance and reduces KV cache size and that's where MLA plus rope comes into the picture. Right? The
58:08
last MLA which we saw that was the basic latent attention mechanism. What it did is we don't need to now save two values
58:15
for K and V. So one two is gone here. We only need to save the latent KV cache. Secondly, instead of saving the keys and
58:23
the values, we only need to save my latent matrix, right? So instead of this
58:28
N into H, we just have DL over here, which is the
58:33
um which is the dimensions of my latent matrix. So this is
58:40
the memory size formula for the multi head latent attention. And we saw that this is significantly lower than the
58:47
multi head attention but it also maintains good performance with MLA plus rope. The thing which we saw today the
58:54
only thing which is modified in this formula is instead of just having the latent CKV which we saw over here CKV is
59:02
being cached that much is fine but along with CKV we are also caching DHR right
59:07
why why we are caching DHR because that's KR and we only need to cache it for one head since it's replicated
59:14
across different heads so the memory size is now DL plus DHR and in the
59:20
original DeepS paper they use DL to be 4 * H where H is the attention dimension
59:25
which they have and DHR they used HX2 where H is the original attention head
59:31
dimension so that's DH so DHR they used H by 2 and DL they
59:38
used 4* DH so if you now divide MHA divided by
59:43
MLA plus rope you'll get that MHA is higher by a factor of 2 into NH divided
59:49
by DL plus DHR Right? So that will be 2 into n into h divided by 9 h by 2. So
59:56
this will give you 4 n by 9 and n which is the number of attention heads which
1:00:01
was used in deepsek that is equal to 128. So in deepsek if you compare multi
1:00:07
head attention with latent attention plus rope you'll see that multi head attention is takes 57 times more memory
1:00:14
requirements as compared to MLA plus rope. So latent attention plus rope actually saves the memory capacity as is
1:00:21
shown over here u by 57 times. So this is 9x2 HL. Where does this 9x2 come
1:00:27
from? So DC is what we are calling DL over here that's 4 * H or 4 * DH and DHR
1:00:35
is DHX2. So overall this comes to 9 DH2. So what you can see over here is
1:00:42
the capability right. uh a multi head attention is a strong performance whereas GQA and MQA don't have good
1:00:49
performance whereas MLA so when they say hours it's MLA plus rope that has a
1:00:55
great model performance plus at the same time they reduce the KV cache by a huge
1:01:00
amount compared to the multi head attention so with MLA plus rope we truly get the best of both worlds we get the
1:01:06
capability plus at the same time we get the KV cache memory reduce uh reduction
1:01:12
I hope all these This formula makes sense to you. Now once you visually understand what is happening on the whiteboard in these computations, you
1:01:19
will know why there is DL plus DHR here which needs to be cached. So Deepseek had this graph where
1:01:26
they actually compared MHA and MLA. Forget about the MO part for now. That
1:01:32
is the mixture of experts. We are going to come to that later. For now just look at the MHA and MLA part. you'll see that
1:01:39
the MLA performance is mostly equivalent or better than the M uh than the MHA and
1:01:47
that's amazing right we actually get a better performance with latent attention while at the same time we save the cash
1:01:54
so here you see KV cache per token in MLA is 34.6K 6k and here it's 860. So
1:02:02
here there is a factor of 25 or 30 and here again there is a factor of 6 or 7.
1:02:08
So we save the memory plus at the same time you get great performance. That's the beauty of the latent attention with
1:02:15
rotary positional encodings. And the reason I believe deepseek added rotary positional encodings is to get the
1:02:21
better performance values which we see over here in all these benchmarks. But they truly achieved the best of both
1:02:27
worlds. they reduced the memory size and got a good performance. In this one lecture I uh we
1:02:33
have covered everything which is present in the deepseek version two and version three paper on multi head latent
1:02:38
attention. I believe I have not left out anything. Now you can truly understand this entire entire diagram. I already
1:02:46
explained you this diagram previously but now you can see these you can understand the shaded portions also. You
1:02:51
only need to cache the KR that two only for one head and the
1:02:57
latenc that's why it's shown with a dash over here and that's why it's shown cache during inference. So you will
1:03:03
you'll understand the schematic you'll understand all of these equations 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
1:03:13
19. So the this I believe until here when you start reading the deepseek paper the architecture part of it is
1:03:20
extremely important but it's also very difficult for people to just read through it you need to make notes and
1:03:25
even when you make notes you need to go deeper into it I hope this lecture serves as a key facilitator for you to
1:03:32
understand everything in the deepseek paper just like I have done with uh latent attention my next target is going
1:03:38
to be a mixture of experts so the mixture of experts is one of another
1:03:43
incredible innovations in the deepseek paper. We are going to go through that and after that we are going to go
1:03:48
through u multi-token prediction. Thanks a lot everyone and this was one
1:03:53
foundational lecture in building and understanding deepseek from scratch. It took a very long time for me to prepare
1:04:00
this lecture but I hope it was worth it and I hope all of you have really understood how latent attention plus
1:04:06
rotary positional encoding was implemented by deepseek. Thanks a lot everyone and I look forward to seeing
1:04:12
you in the next lecture.




















