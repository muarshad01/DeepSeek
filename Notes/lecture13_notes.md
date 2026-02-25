### MLA Code

* 15:00


new token will come in and the embeddings for the new token token will be given by this. Okay, this will be my
15:06
new input token and then my goal is to pass this input token through my uh
15:12
inference pipeline and to predict the next token. All right, so let's get
15:18
started. Um before we get started, we need to decide a few things and
15:24
simultaneously I'll walk you through the code. So the class which I'm defining is
15:30
called roless MLA. The why we are using the word roless is because an advanced version of MLA use rotary uses rotary
15:37
positional encodings which we are not using here. That's why I've given the name roles. Don't be confused by this
15:43
name. So we have to decide the D model which is the embedding dimension of our
15:49
model the number of heads and then DH is the head dimension which is D model divided by number of heads. Great. Now
15:57
let's define a few matrices. Let's go to our figure and let's actually see what
16:02
are the matrices which we need to define. Uh for the sake of simplicity,
16:07
what I'm going to do is that I'm going to copy this entire thing and I'm going to bring it down so that we don't have
16:13
to go um we don't have to scroll up every single time. Okay. So let's see what all are
16:20
the matrices which I need. First I need WQ, right?
16:25
Um let me remove the yeah first I need WQ which is the
16:32
weight matrix corresponding to the queries the trainable weight matrix. What are the dimensions of WQ? The first
16:39
dimension of WQ will be given by the input embedding dimension of the model.
16:45
So that will just be D model. And the second dimension of WQ is the output
16:50
embedding dimension. And in the code we are going to choose this to be same as D
16:56
model itself. In this case I've chosen it to be four. But in the code we are choosing this output dimension to be the
17:02
same as the input dimension of the model. So I defined this trainable
17:08
weight matrix to be having dimensions of D model, D model. And the way I have
17:13
initialized this is a neural network linear layer and I put bias equal to false. That's a much more efficient way
17:20
to initialize it compared to NN.parameter because weight
17:25
initializations are much more robust when we use NN.linear. The second matrix which we
17:31
need to initialize is WDKV and this is the most important matrix because when we multiply my input
17:38
embedding with this matrix that ultimately gives me my latent matrix. What are the dimensions of WDKV? The
17:46
dimensions are um the input. So what is this 8 here? That's the input embedding
17:51
dimension and this four is my latent dimension. So WDKV has D model which is
18:00
the input embedding dimension and the latent dimension. Then I have to define W UK and
18:06
WUV. What are the dimensions of both of these? The first dimension is the latent
18:13
dimension for both of these and the second dimension is the output dimension of the model which is same as D
18:20
model. So I define W UK and W UV and then there is this WO here which is the
18:27
output projection which takes me from my context matrix to my logit matrix. Don't worry about this right now. We are not
18:33
going to be too concerned about this for the purposes of this code.
18:39
And then I have also defined a layer normalization. So we'll be passing the we'll be passing our uh latent vector
18:46
through a layer normalization layer. All right. So these are the matrices which I need to define and you
18:52
will see that I have defined them here also. And during inference these matrices are fully trained which means
18:58
they won't be updated anymore. I have defined WQ which is now D model by D
19:03
model 8 by 8. I have defined WDKV which is now D model multiplied by my latent
19:09
dimension. Then I have defined W U K which is the latent dimension multiplied by D model. I have defined W UV which is
19:17
the latent dimension multiplied by D model. And finally I define WO which is the output uh projection weight matrix.
19:24
All right, great. What we have to do next is that we have to uh define the
19:30
forward method. This is where all the magic happens. So let's start going through it step by step. First I have
19:37
defined my input. So if you see X X dot size right. So if X is my new token
19:43
which is coming in um the batch will be equal to 1. The context size will be
19:49
equal to 1 because I'm just looking at one token and the number of dimensions will be equal to 8. So the size will be
19:55
1a 1 8. In general this method can take input of any batch any context size and
20:01
any dimension. So I can even pass in this entire input sequence which has
20:07
which would have the dimension of if the batch is one then I have five tokens
20:12
here and then each token has eight dimensions. Okay that's why there are these three dimensions over here for now
20:19
you can imagine one token which is passing and it's 1a 1a
20:25
8. All right. So this is my x dot size. Then what we have to do is that remember
20:31
we have to define this absorbed query matrix. So we have to define this
20:37
absorbed query matrix which is WQ multiplied by W UK
20:43
transpose because I'm going to ultimately use this product directly. So I have to define this matrix now and
20:50
that's what we are going to do before we actually compute the latent KV cache.
20:55
What we are going to do is that we are going to define the absorbed query matrix. Um the absorbed query matrix
21:01
will be a product of WQ multiplied by W UK transpose. So WQ is 8 by8 and W UK
21:09
transpose will be 8x4. So the resultant will be an 8x4 matrix. So you'll see in
21:14
the code what we are going to do is that first we are going to multiply WQ * W
21:21
Uranspose. So the way this torch.matmo works with the weights of these neural
21:27
networks are that when we write torch.mmet it actually multiplies wq with w
21:34
transpose. So this line of code is performing the same step as what I've mentioned here. Wq is multiplied with wk
21:41
transpose and then I get my absorbed query ve query matrix. Okay.
21:48
Um now what happens after this point is that which we have not seen in
21:54
yesterday's lecture is that there are multiple attention heads which are there right. So the absorbed query matrix
22:00
needs to be split among the different heads. So the first attention head will have access to half of
22:05
this. This will be head one and the second attention head will
22:11
have access to half of this which is going to be head number two. So we are going to group this by the number of
22:16
heads. So the dimensions here were 8x 4. Now the dimensions are going to be 2a 4a
22:23
4. Why? Because this is going to be my head number one and this is going to be my head
22:29
number two. Um so actually let's call this head
22:35
number zero and let's call this head number one because there are two heads. So the first head will have 4x4. The
22:41
second head will be 4x4. That's why this will be 2a 4a 4. Right? When we look at
22:47
each head, the number of rows um are equal to the head dimension and the number of
22:55
columns here are equal to the latent dimension. All right, great. Now what we can do is that
23:03
let's see first of all whether this the same thing which is being implemented in the code. So in the code what we are
23:09
doing here is that absorbed dot view n heads which means here we are telling
23:16
that we have to use the absorbed matrix and we have to group it with the number of
23:21
heads. So the dimensions of the new matrix will be batch size number of
23:26
heads um actually batch size won't be here it will directly start with number of heads
23:33
head dimension and the latent dimension. So this will be number of heads, head dimension and the latent dimension. The
23:39
way to visualize this is that now we have one matrix for head the first head and we have another matrix for the
23:46
second for the absorbed query. Next what we have to do is that
23:51
we have to uh we have to first find our original latent cache. So this is an
23:59
important step. Remember what I mentioned here the X into WDKV will be
24:04
cached right and previously there were five tokens which have which I've entered. So we will have a cache which
24:10
is corresponding to X which is 5 by 8 multiplied by WBKV which is 8x4. So our
24:17
previous cache will be 5x4. The first row corresponds to the the second row
24:22
corresponds to next day is Nbrite.
24:27
This is my previous latent KV cache. The dimensions here are four because my KV dimension is four and I have five tokens
24:34
which are being cached whose values are being cached. Right? So this is my previously computed KV cache. And what
24:41
I'll do now is that I have this new token which has entered. Right? I have this new token which has entered. I will
24:48
compute the latent KV cache for this new token. So the way I'll do that is that
24:54
I'll multiply this new token embedding with WDKV and this new token has a size
24:59
1x 8 because it's an 8 dimensional vector and WDKV is 8x4. So once I do
25:04
this multiplication I'll get my new latent KV cache vector that's 1x4 I'll
25:11
apply layer normalization to it. So that's my new latent KV vector. This new
25:18
vector I'll append to the previous cache. So this new vector will be appended to the
25:24
previous cache. This new vector will be appended to the pre previous cache. And then I will get my new latent KV cache.
25:32
Take a look at the dimensions over here. Now this will be 6x4. Why? Because I have six tokens. The next day is right
25:38
and my new token which is here. So this is 6x4. Now that's how the cache
25:44
actually goes on increasing whenever new infer token is appended right and the reason I have four dimensions over here
25:51
or four columns is because the latent dimension is equal to four. So remember
25:57
what we have done here is that we have first looked at our previous cache and we have computed the cache corresponding
26:04
to my new input uh token and then I have just appended it to the previous cache
26:09
and then I get my updated KV cache. Why am I doing this? The reason I'm doing this is because to get the attention
26:16
scores or to get the attention weights remember what I have to do to get the attention scores I have to multiply this
26:23
absorbed query with the updated KV cache. That's why I'm I need to find my
26:30
updated KV cache. All right. So until now what we have done on this whiteboard is that we
26:36
have found the latent KV representation for my new token and we have not just
26:42
done that we have also appended it to our old KV cache and we have got the new cache. Let's see how this that is done
26:49
in the code right now. Um so see what we are doing here is that first we are finding the latent KV
26:57
vector for the new input token. So if the new input token is X, I'm going to I'm going to multiply it with
27:03
WDKV. Uh and then I'm going to apply layer normalization. So this self.l
27:08
essentially just does layer normalization. So this one step or this one line of code is actually doing two
27:15
things. I'm multiplying my input embedding with WDKV and then I'm applying layer
27:22
normalization. So that gives me my new latent KV vector. And since we are caching KV cache is not none. So what we
27:29
will do is that we'll take our new KV vector and we will append it to my previous cache. This is how I get my new
27:36
KV cache, right? Um and the dimensions of this are S total which are the number
27:42
of total tokens inferred up till now. And the latent dimension is the number of columns. B is just the batch
27:49
size. So until now what I've done is that I've got my new um or I've got my
27:55
updated latent KV cache. So let's move to the next step now. All right. So in
28:01
the next step, what we have to do is that we have to slowly start computing our attention scores, right? And the way
28:08
to do this is first we need to compute our absorbed query. Which means what we'll need to do is that we'll need to
28:14
multiply X which is my input multiplied by my this absorbed weight matrix which we
28:21
have already computed. Remember here what we have done is that we have already computed this absorbed K matrix
28:28
right which is WQ multiplied with W Uranspose. So this this we have already
28:35
computed we now just need to multiply X with it. But remember what we did after WQ multiplied by W UK transpose is that
28:43
we split it into two heads. So if you remember what we have done here this matrix we have split it into two heads
28:49
head zero and head one. So to multiply my input embedding vector with these two
28:55
heads. Now what I'll have to do is that I'll have to split my new tokens input into two heads. So the new token input
29:03
was an 8 dimensional vector which was this. I'll split it into half. So half
29:08
the first half will correspond to head number one. The second half will correspond to head number two. Right? So
29:14
for the first head what I'm going to do is that I'm going to multiply the first half of the input multiplied with the
29:21
first head absorbed query matrix. So that will give me the absorbed query vector for the first head. This will be
29:28
1x4. This will be the absorbed query vector for the first head. So what I'll do is that for the second head I'm going
29:34
to do something similar. For the second head what I'll do is that I'll take the second half of this input. I'll take the
29:41
second half of this input and I'll multiply with the second absorbed head or second absorbed query. So that gives
29:48
me the absorbed query vector for the second head. This multiplication essentially what we are doing here is
29:54
that we are just multiplying we are just multiplying x with this wq * wrpose. But now we are
30:02
splitting it into two heads, right? We have the first head and we have the second head. So the first head will
30:08
contain So the first head will contain the product of half of this input
30:13
embedding multiplied by the absorbed qk0. Absorbed qk0 is this first the
30:20
absorbed matrix corresponding to the first head and for the second head I'll multiply the second half of the input
30:27
embedding multiplied with absorbed qk1. Okay. So now I have the absorbed
30:32
query vector for the first head and the absorbed query vector for the second head. Once I have the absorbed query
30:38
vector to get the attention scores all I have to do is that I have to multiply the absorbed query vector with my
30:44
updated cache which we have already calculated. So to get the attention scores what we do is that we take the
30:52
for head number for the first head we take this absorbed query vector and we multiply it with the cache
30:59
transpose. So remember we calculated the new cache right. So to get the attention
31:06
scores we have to multiply the absorbed query vector with the new cache transpose. So this is for head number
31:12
one and the attention scores for head number one is a 1x 6 vector. Why is it 1x 6? Because the absorbed query is 1x4
31:20
that will be multiplied with the cash transpose which is 4x 6 because cache is
31:26
originally the new cache is 6x4. So the cash transpose will be 4x 6. So 1x4 * 4x
31:34
6 gives me the attention scores for head number one that is 1x 6. Similarly for
31:39
head number two I multiply the absorbed query vector multiplied by the new cache
31:44
transpose that gives me the attention scores for head number two which is 1x 6. Let's do a sanity check to consider
31:51
why the attention score has six values. The reason the attention score has six values is remember when I append the new
31:58
token. So the first five tokens are the next day is bright and then I have my
32:07
new token. Right? Essentially in the attention scores what we are doing is that we are taking the new token and we
32:13
are computing its attention score with itself and all the other tokens. That's
32:18
why there will be six values because there are six tokens right now. So the attention scores for
32:25
this new token has six values for head for the first head and for the second head.
32:30
Let's see how that is implemented in the code. So in the code what we are going to do is that in this part this is the
32:36
part where the input is split into two halves. Right? This is the part where I take my input and I split it into two
32:43
heads. Head number one and head number two. Then in this part what I'm doing is
32:48
that first I multiply the uh the absorbed which is WQ * Wrpose
32:55
with the first half. So this PMP are the absorbed query vectors which I get for my all the headers. So this has been
33:03
mentioned over here. So this PMP is this which are the absorbed query vectors
33:08
which I get after multiplying the input for that head multiplied by the absorbed values for that head. Um and then once I
33:16
get these tmp values to get the attention scores all I have to do is that I have to multiply these uh these
33:23
values multiplied with the new cache transpose. Right? So here what is done is that I get these values and I
33:30
multiply it with the new cache transpose. So this is the absorbed query
33:35
vector and I multiply it with the new cache transpose. So this gives me my attention scores values for head number
33:42
one and for head number two. Okay, once I have the attention scores for head one
33:47
and for head two, what I do is that I will do the uh scaling by the square
33:53
root of the keys dimension. I will do the scaling by the square root of the keys dimension and I'll
34:00
um apply soft max to get the attention weights. So I get the attention weights for head number for the first head and I
34:08
get the attention weights for the second head. This step is shown in the code in
34:13
this part. So I take my attention scores and I have my head dimension. I divide
34:19
by the square root of the head dimension. Um then what I do is that I just do a soft max here. Okay, I get my
34:26
attention weights until this point. Now once we get the attention weights,
34:31
we have to find the context vector. Right? For that we need the values matrix. To get the values matrix, we
34:37
just have to take my updated cache value and multiply it with WV. Why? Because if
34:42
you see this to get the values matrix we just have to multiply my updated cache value with w and then I'll get my values
34:50
matrix. This is exactly what we do. We have already computed this updated cache right that's a 6x4 matrix which has
34:56
already been computed over here um my updated cache which is 6x4. So I
35:02
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




