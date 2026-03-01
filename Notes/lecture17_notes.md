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
                        &= \underbrace{Q_CK_C^T}_{Ratain ~old ~magic ~MLA} + K_R^T
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
|$$d_{C}^{'}$$||
|$$d_h^R$$||
|$$n_h$$||
|||

we do is we take this same matrix and we extend it across two heads. So I copy it across the two heads which I have and
30:11
that will ultimately give me the dimensions of 4, 8 and that is my KR. So
30:16
remember I told you K R will be of a dimensions of number of attention heads into BHR. So now it becomes of the
30:24
number of dimensions 8 which is BHR into NH. So what we do is that we actually
30:30
use the same W KR across all the heads. Uh as is written over here the same key
30:36
is shared across all the heads. So we use the same W KR and this is the and
30:41
then we apply rope and then this same key matrix is shared across all my heads. So we actually repeat this across
30:47
two heads and ultimately we get the KR matrix which is 4x8. The way KR is
30:53
different from KC is that the dimensions. So here KC has the dimension equal to D and KR has dimensions equal
31:00
to DH. D KR has the dimension equal to DHR into NH which here I have chosen to
31:07
be equal to 8 but it can be even different than 8 if needed. Another difference here is that W UK actually is
31:15
different for different heads, right? We are not sharing anything across heads over here. But here we are sharing the
31:22
values of the keys across the two heads. So that's how the KR matrix is computed.
31:27
And the big another big difference is of course KR has a rotary positional encoding included into it.
31:34
Now we see how the queries is computed with rotary positional encoding encoded or with rotary positional encoding
31:41
injected. So if you see on the left hand side we down projected and up projected right the same thing is going to happen.
31:47
We are first going to down project WDQ and this is the same dimension DCash
31:52
which we saw over here DCash. So we are going to take my input 4x8 and we are going to down project it uh by
31:59
multiplying it with wdq. So this gives me my CQ matrix which is 4x4. That's the
32:05
same CQ matrix which we saw over here. Right? So similar to W KR we have WQR.
32:12
Now and note the dimensions of WQR. The dimensions of WQR is DHR into NH. So
32:21
this is not being shared across heads unlike W KR. So that's why even in the
32:27
description we have separate we use additional multi head queries which are separate for different heads and a
32:33
shared key. So the key thing which deepse actually implemented is that uh
32:39
when they actually constructed these uh matrices with rotary positional
32:44
encodings injected the queries matrix were not shared across different heads. They were different for different heads.
32:51
So this CQ was multiplied by WQR whose dimension was DHR into NH already. So we
32:58
don't need to multiply across heads in this case. So CQ you multiply with WQR
33:03
and then you get 4 by 8. Why four? Because four are the number of tokens over here. Then you apply rotary
33:09
positional encoding and then you get your QR matrix. See now you have the QR matrix and KR both have the size 4x8. So
33:17
the dimensions of both are DHR into NH. But the way we get these is different
33:22
right because in the KR we actually expand across the heads. So we share these values across heads. But in the
33:28
case of the queries matrix there are different for different heads because the dimensions of W KR the number of
33:34
columns is equal to 8 whereas the number of columns of W KR is equal to
33:39
four. So take a look at the dimensions which I'm using here. Right? I'm using
33:44
how many dimensions I'm doing? D which is my original dimension. I'm using DC
33:49
which is my latent dimension. I'm doing DC dash which is my queries down
33:54
projected dimension. Then I have something which is called as DHR. I have
34:00
something which is called a DHR which is my rope dimension. And I have NH which is number of attention heads. And I have
34:06
DH which is my original head dimension. So all of these variables which you are
34:12
seeing on the whiteboard right now. You will find all of these variables over here. You'll see DHR, you'll see NH,
34:20
you'll see DC dash. So I'm using the same variable so that after this lecture you can go back and completely
34:26
understand what is written in the code over here. Um, okay. So let's see what all we have covered so far. Right? So on
34:33
the left hand side of the screen, we computed KC, VC and QC. So if you see
34:39
the paper now we have KC we have QC we
34:44
have K we have KC VC and QC which have been computed and on the right hand side
34:50
of the screen right hand side of the screen we have KR and we have QR which
34:55
have been computed. So on the right hand side of the screen we have um QR which
35:02
has been computed and which have we have KR which has been computed. So we have finished step number nine, step number
35:09
10, step number 11, step number 12, 13, 14 and 15. So what happens in step
35:16
number 9, 10, 11 and 12, 9, 10, 11, 12, 13. That's everything on the left hand side. We first get the CKV matrix, we
35:23
get the KC and the VC. That's what we started with. Remember, we first get the
35:29
uh we first get the CKV matrix, then we get KC, and then we get VC. We saw that.
35:34
Then after that in equation number 12 and 13 we get CQ and we get QC. So down
35:40
projection and up projection of queries. Here is what we see. We get down projection of queries to get to give me
35:47
CQ and then we get up projection of queries to give me QC. Great. So here we have exactly mapped equations 9, 10, 11,
35:55
12 and 13. Then we go to equation 14 and 15. In equation number 14, what we do is we
36:02
do the QC part, right? We start with CQ and we multiply with WQR. What's being
36:08
shown over here? We start with CQ and we multiply with WQR and then we apply rotary positional encoding to it. That's
36:15
how we get the uh that's how we get the QR. And then in
36:21
equation number 15, we we use uh we multiply the input with W KR and then we
36:28
expand it across the heads. That gives me my KR. Okay. Um so we have covered
36:35
everything until steps 9 9 10 11 12 13
36:41
14 15. Now let's come to step 16 17 and 18. What happens next is that we have uh
36:48
uh we have Q KC QC VC where no rotary positional is encoding is applied and we
36:54
have KR and QR right. The next step for us is to find the attention score. So let me rub this a bit so that um the
37:03
next step for us is to find the attention scores. Right? So the way we do it is that we will multiply the
37:09
queries multiplied with the keys transpose. And to get the queries what I do is I have my QC matrix which is the
37:15
4x8 matrix. Correct? Uh that's my QC matrix and I have
37:21
my QR matrix which is my 4x8 matrix. So I'm
37:26
going to concatenate these two matrices together. I am going to so I have this QC over here and I have the QR over
37:33
here. I'm going to concatenate these two matrices together. So QC and QR and
37:38
similarly I'm going to concatenate my KC matrix and my KR matrix and my KR
37:44
matrix. So I'm going to concatenate KC and QC and QR to give me my full queries matrix. I'm going to concatenate KC and
37:52
KR to give me my full keys matrix. And I'm going to multiply this entire queries multiplied by this entire keys
37:58
transpose. And then I can split it out mathematically, right? I can split out QC multiplied by KC transpose transpose
38:05
plus QR multiplied by KR transpose. Uh so QC when I multiply QC
38:11
and KC transpose I'm not doing rotary positional encoding at all. So I can do the absorption trick and get my
38:17
attention scores which is a 4x4 value since there are four tokens and when I do QR into KR transpose it's again um so
38:25
QR is uh QR is 4x 8 and KR transpose will be
38:30
8x4. So this will be 4x4. So then I have one attention scores which is 4x4. Another attention scores which is 4x4.
38:37
So I add these two attention scores together to get my final attention score. Then I will scale it. I will
38:44
scale it by square root of dh plus dhr where dh is my original dimension head
38:51
dimension equal to 4 and dhr is my rotary rotary dimension which was equal
38:57
to four. So I I will scale it by divided by square root of 8 so that the quer is
39:02
multiplied by keys transpose variance is constrained within one. Then I apply soft max. I get my attention weights.
39:09
These attention weights will then be multiplied with VC. Remember VC is what we already computed to be 4x8. So this
39:16
4x4 matrix will be multiplied with the 4x8 and then I get my context vector matrix which is a 4x8 matrix. That's
39:22
all. So in the schematic so or in the equations you see we first in this step
39:30
number 16 we concatenate QC and QR. In step number 17 we concatenate KC and KR.
39:35
And in step number 18 what we are doing is that we are doing multiple things here. Actually we are scaling it by
39:42
square root of DH plus DHR. We are applying soft max. So this steps this step gives us the attention weights
39:49
which we saw over here. the attention weights when we scale it by square root of DH plus DHR and apply soft max and
39:56
then this multiplying with the value vector here that gives me my context vector matrix which they call O the
40:04
context vector matrix then ultimately goes through my output head and that gives me my logit matrix which they have
40:09
denoted by UT over here but what we have done on the whiteboard right now is I have visually shown you uh just on this
40:18
simple whiteboard that how We can go from step number 9 to step number 10 11
40:26
12 13 14 15 16 17 18 and 19. So this is
40:34
the entire multi head latent attention logic which was implemented in just one
40:39
page in the deepseek paper. It's very very very difficult to understand this without writing it down on a whiteboard
40:45
and for me also it took around one and a half months to construct this lecture because it was quite hard to figure out
40:51
these dimensions honestly and some minute details like the keys are shared in the rotary part but the queries are
40:58
not shared. Uh so to truly understand this lecture you also need to understand rotary positional encoding which we
41:04
covered in one of the previous lectures. But through this trick of decoupling, we can use our absorption trick on the part
41:12
which does not use rope and we can treat the other part separately. Then we get the attention scores, attention weights,
41:18
context vector matrix and that's the whole flow. So if you scroll back up and see
41:23
the diagram which they have given um this is the multi head latent attention
41:30
diagram. Right now you can begin to appreciate this diagram a lot better. You start with the input embedding which
41:36
they which they denoted as HT. Then first let's let's see what we did on the
41:42
U query side right on the query side there are two parts. There are QC and QR
41:49
which you will ultimately concatenate. On the key side there is K KT and there is uh there is KR and there is KC which
41:56
you will ultimately concatenate. That's what's shown with this concatenation. Right? uh and in the query part there is
42:03
one more thing in the query you do the down projection. So first you get the CQ and you apply rope to CQ to get the QR
42:10
part. So if you remember what we did in the in here is that we apply rope to CQ
42:17
uh to get the QR. So to the down projected query. So that's what's shown
42:23
in this part. We first get the down projected query. We apply rope by multiplying with W QR and and then we
42:29
get QR and then we concatenate QC and QR on the right hand side. What shown is
42:34
that we first get the latent CKV matrix. That's how we get the KC and the VC. Right? This was what we saw in the left
42:41
hand side of the board. We get the latent CKV matrix and then we get the KC and the VC.
42:47
uh but one more thing which is shown here is that we take the original matrix and we apply rope to get the KR and then
42:54
we concatenate KR and the KC to give me my concatenated K matrix and then we
43:00
apply latent attention. So here this part the absorption trick the QC the KC
43:05
and QC the QC and KC the absorption trick works and the other part QR and KR
43:11
is treated separately and then we get my final context vector matrix which is denoted as U over here. So one thing you
43:19
might be thinking about is what is this dash over here right which says cache during inference and it seems to imply
43:26
that we only need to cache two things the latent KV matrix and the KR matrix.
43:32
So let's see in the schematic in the schematic I showed you that we need to cach the latent CKV matrix. This much
43:38
seems fine because even in the original latent attention we only cached the CKV matrix and since the left side of the
43:45
board is the original latent attention it makes sense that we only cached the CKV but in the right hand side if you
43:51
observe carefully I have shown a black box not these two black boxes but I'm
43:57
talking about this black box over here I've shown black box a dotted black box
44:03
for the KR matrix because now along with caching the CKV matrix mat we also need
44:08
to cache the KR matrix because we have additional computations over here. So in
44:13
the next part of this lecture we are going to see what happens exactly when a new token comes in during inference and
44:20
why do we need to cache two matrices. We need to cache the CKV matrix and we also need to cach the KR matrix. So let's see
44:27
that right now. So let's say a new token comes in which is bright. So imagine that we are
44:34
doing the inference and the next day is and the token which is inferred is bright right so now when bright comes in
44:41
at as a new token let's see how we predict the next token through this entire MLA plus rotary positional
44:47
encoding framework right u so we have to go through two paths the attention
44:53
scores path one and the attention scores path two what is path one path one is
44:59
basically no rope and path two is where we are going to apply rotary positional
45:04
encoding. So as we saw before path one is QC into KC transpose and path two is
45:10
QR into KR transpose. So let's go to path number one right now and let's then go to path number two. Remember in path
45:17
number one we are going to use the absorption trick. So path number one proceeds in a
45:24
very similar manner as our basic latent attention but with the down projection and the up projection of the queries.
45:31
Right? So what we are going to do now is that uh whenever a new token comes in we
45:37
first find its input embedding and then instead of earlier if you see earlier what did we have in the absorption trick
45:44
in the absorption trick earlier we had WQ into W UK transpose but now WQ won't
45:51
be there instead of WQ we have WDQ and WUQ which are the down projection and
45:57
the up projection these will be absorbed so in our previous implementation this
46:02
was just WQ. So now my QC which is my absorbed query vector for bride will
46:08
will be X which is the input token only for bride. So that's 1x 8 multiplied by
46:13
WDQ the down projection query multiplied by W UQ the up projection query
46:19
multiplied by W UK transpose. So again pay careful attention to dimensions. This will be equal to D. WQ will be my
46:27
down projection. So take a look at the dimensions over here. My down projection will have DC
46:33
dash DC which is equal to 4 and my up projection will again again transfer it
46:38
into D and my UK transpose will be uh take a look at the W UK dimensions.
46:46
The W UK this is my WUK transpose. So this is 8 * 4. This four
46:54
is my latent space which is my DC and my 8 is equal to D. So when you multiply
47:00
all this you get the absorbed query vector for brite which is now going to be 1a 4. And then what you need to do is
47:07
you need to compute the latent KV vector for Bright. So this is going to be the input embedding for Bright 1X 8
47:14
multiplied by WDKV right and that gives me the new CKV vector 1x4. And remember
47:20
I'm caching this I'm caching my uh I'm caching my CKV right so the original CKV
47:28
which was there um that was 4x4 and now I have an additional 1x4
47:35
matrix based on this new token which will be updated to my latent KV cache. So my new KV cache will be a 5A 4
47:42
matrix, right? And then what I need to do is that to get these attention scores, I just need to multiply my
47:49
absorbed query vector into my updated KV cache. So remember the power of the
47:56
absorption trick. We are again using it over here. We only need to cache the latent KV. We don't need to cache the
48:01
keys or the values separately at all. Um so we are going to use my absorbed query
48:06
which is 1x4 1A 4 and we are going to multiply it with the updated KV cache transpose which is 5a 4 transpose. So
48:15
multiplication of this will give you an attention score for bright which is 1a 5. Why are there five? because now I
48:22
have the next day is bright and the attention score I'm taking with bright
48:30
as the query with itself and with all the other tokens. So that's why it's 1a 5. If you feel that I went a bit fast
48:37
through this part, it's because we have already covered this in the latent attention. The only difference is that
48:43
instead of WQ, we have WDQ into WUQ. So please go back and revise the latent
48:49
attention lecture if this part is not very clear to you. So we went we
48:54
calculated the attention scores through path number one right now which was a 1a 5 vector and here we only needed to
49:01
cache the latent kv that much is good now we don't stop here right because we
49:07
need to compute the attention scores through path number two as well which is my which is where my rotary positional
49:13
encodings will be injected. So here I have to compute QR multiplied by KR transpose. The way it's done is that
49:20
remember first we'll compute KR. So we multiply the input with W KR. So the input which is 1x 8 which is
49:28
the token embedding for bride is multiplied with W KR. It is 8x4. Remember this is DHR which is equal to
49:35
4. The rotary each attention head dimension in my rotary space. So this product will be 1x4. I will apply the
49:42
rotary positional encoding to it which will preserve its size. So my keys vector will be 1x4. But this is only for
49:49
one head. So I'll need to expand it across two heads so that it will be 1a 8. And one more thing which I want to
49:55
mention here is that we are going to cache my KR matrix. Right? So if you see
50:00
the KR matrix which was computed earlier, this had a size of 4A 8 for the four tokens and we have cached it. And
50:08
now I'm going to append my new KR vector to this cache. So I'm going to append my 1x8 vector to the 4x8 cache. And that
50:16
gives me my updated KR cache which is a 5x8 matrix at the moment. Okay. So this
50:22
is my KR but I still need to compute my QR right. So to compute QR remember the
50:28
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














