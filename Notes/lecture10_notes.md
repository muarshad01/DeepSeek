
***

* 10:00

#### How to solve KC cache memory problem?
1. Multi-Query Attention (MQA)
* What if all the attention heads share same Key & Value matrics?
3. Group-Query Attention (GQA)

***

* 15:00

head four all have the same values such as like this if you look at the values Matrix
15:51
what if head one head 2 head three and head four all have the same values such
15:56
as I'm showing right now so that actually means that if you look at the trainable Keys Matrix and
16:03
trainable value Matrix here also head one head two head three and head four
16:09
will have the same values so among these different heads the values are same
16:14
that's why I've shown it with a different color within one particular head the values can change but if I
16:19
showed them with a different color it would be too many colors so that's why for the sake of Simplicity within one
16:24
head also I've shown with the same color but the real point I'm trying to illustrate here is that for head one
16:30
whatever the values are there for head one it's the same for head 2 head three and head four and that's exactly the
16:36
same for the values weight Matrix whatever values we have for the head one it's the same for head 2 head three and
16:43
head four then if you think about it then I just need to cach it for one head right
16:49
I just need to cach the keys and the values for one head I don't need to cash for all the other heads because they are
16:54
the same values so why should I store more things in my memory if uh all of these head two head three and
17:02
head four are the same values so that's why what happens in multiquery attention
17:07
is that if you look at the size of the KV cache it depends on the number of attention heads right now this is taken
17:13
out of the picture completely this is taken out of the picture completely because we get rid of
17:19
uh we get rid of the differences in the heads which means that now I only need
17:25
to store the keys and the values for one head so so in uh in the case of
17:30
multiquery attention the size of the KV cache size or the size of the KV cache suddenly reduces by a factor of
17:38
N and if you have 32 heads or if you have 128 heads like in the case of deep
17:43
seek deep seek has 128 heads so multiquery attention reduces the KV cach
17:48
size by a factor of 128 that's a huge reduction that much is the reduction in
17:54
the memory space if you use multi querer tension so let me re uh revisit what I
18:01
just mentioned what I just told you is a simple trick I told you that instead of
18:06
having different WK and WV values for different heads what if you just have it
18:12
for one head and then you create copies for head 2 head 3 and head
18:17
four so if you create copies in WK and WV even in the keys Matrix what will
18:23
happen is that you just get the keys uh Matrix for head one and you create
18:28
copies for head 2 head three and head four so even if you create the values
18:33
for head one and you create copies for head two head three and head four that is what is actually meant by what if all
18:40
attention heads share the same key and the value matrices right now what this ultimately
18:46
leads to is that it leads to the fact that the size of the KV cach reduces by factor of n because I don't have to
18:53
store the keys and the values for all my heads right now I just store it for one head that's it and then when the time
18:59
for inference comes I just use the same values across different heads that reduces the KV cach size now
19:07
one thing to keep in mind here is that the queries if you look at the query vectors for the different attention
19:13
heads q1 Q2 Q3 and Q4 those are different I'm just saying that in
19:18
multiquery attention the keys and values share the same values across heads but if you see for head number one for head
19:25
number two for head number three and head number four I have marked the query vectors with different colors the reason
19:31
I've marked them with different colors is because they are different I'm not making the queries same across different
19:39
heads if you at any point you get confused how to interpret the rows and the columns here just remember that in
19:45
the queries keys and the value Matrix the rows are my number of tokens the
19:50
next day is bright and the column columns are my
19:56
total dimensions which were eight but now they are just split into four attention heads so each attention head
20:02
will get a dimension of two the queries so each query still has its own values
20:07
like in multi-ad attention but the but but the keys and the value vectors are
20:13
the same across the different heads now think about this for a moment
20:18
right if you are reducing by factor of n deep seek has 128 head so multiquery
20:23
attention reduces the KV cach size from 400 GB which we had calculated earlier
20:28
for deeps remember the KV cach is 400 GB and now it's reduced by 128 to 3 GB
20:35
that's incredible right um that's a huge reduction in the KV cach size so then
20:41
you might be thinking that wow this is great reduction so why doesn't everyone just use multiquery attention because
20:49
multiquery attention also comes with a dark side and I want you to pause this video for a moment here and think about
20:56
what the dark side might be the hint which I'll give you is think about why we started doing multi-head attention
21:02
right what is the advantage of multi-head attention and uh what are we losing
21:08
out by uh essentially doing multiquery
21:15
attention Okay so let me give you the answer if you think about why we started
21:20
multi-head attention the main thing which we wanted to capture was different perspectives through
21:27
different heads correct uh so let's say if you have a statement like this the artist painted the
21:32
portrait of a woman with a brush it can be either something like this the artist painted the portrait of a woman using a
21:39
brush or it can be something like this the artist painted the portrait of a woman with a brush in her hand it can be
21:46
either of these two if you just use the self attention mechanism it can only capture a single perspective it cannot
21:53
capture multiple perspectives and that's why we had started uh understand in
21:58
multi-ad attention so remember the purpose of multi-ad attention the purpose of multi-ad attention was since
22:06
the queries the keys and the values are split into multiple heads ultimately
22:11
we'll get um an attention weights Matrix for head one we'll get an attention
22:17
weights Matrix for head 2 then we'll get a context Vector Matrix for head 1 and head2 and ultimately when we merge these
22:24
two context Vector matrices the first attention head will Capt capture one perspective my second attention head
22:30
will capture another perspective so the idea of having different attention heads
22:36
was to capture different perspectives now if you suddenly say that my keys my
22:42
K1 and K2 will be the same that's what you are saying in multiquery attention right you're saying in multiquery
22:47
attention that my K1 and K2 will essentially be the same they won't be different values my V1 and V2 will be
22:53
the same they won't be different values what that does is that that red reduces
22:59
the amount of different perspectives which we can capture across these different heads so that means the
23:06
performance of our language model wouldn't be as good as it was with multi-head attention the power of
23:11
multi-head attention was that K1 and K2 are different V1 and V2 are different so
23:16
we had given the each attention head the provision to capture something completely
23:22
different but now if you're making the keys and the Valu same across all the attention heads then you restricting the
23:29
power of the heads right you're restricting the diversity which different heads can capture because they
23:35
all share the same keys and the values although the queries they share are different the queries they share are different so the attention scores will
23:42
be different that's fine but still you are keeping the keys and the values to
23:47
be the same across different heads so you are kind of restricting the amount
23:52
of diversity which you can capture through different attention heads that's one of the huge drawback of multiquery
23:59
attention although it is saving my memory it is we get a significant
24:05
performance degradation and that is not what we want deeps models are hugely
24:11
performative they are great at performance they are great at capturing context so of course this is not what
24:17
deeps implemented uh that's the main disadvantage of multiquery attention
24:23
because we use the same keys and the values across different heads we we get a severe performance it memory is saved
24:30
that's good so we counter the Dark Side of KV cache we save the memory but mqa
24:36
introduces its own dark side and we get a severe performance degradation what I'm going to do right
24:43
now is that after this point I hope you have understood the concept of multiquery attention it's positives and
24:48
its negatives I'm going to now take two models two language models one which
24:53
implements multi-ad attention one which implements multiquery attention the we are already three Trend models and we
25:00
are going to visualize the key value matrices in these heads we are also going to compare the inform the
25:07
inference performance of these models so let's get started with that now all right so the models which I'm choosing
25:14
for this comparison is that we are choosing a gpt2 model which implements standard multi-ad attention and we are
25:21
going to select the Falcon model so if you search Falon uh llm
25:29
you'll see that falicon is a series of language models which are pretty popular they are open- Source models available
25:36
through hugging phas and we are going to use that variant of the Falon model which uses multiquery attention so if
25:43
you see I'm going to code right now I'll share this file with you we are going to use the gpt2 medium model for our
25:49
multi-head attention purposes that's a 355 million parameter model and we are going to use a falcon 1 billion
25:56
parameter model uh which uses multiquery attention which means it uses the same key value matri across all the
26:03
heads now uh I'm running this on an a100 GPU but if you are on Google collab and
26:10
if you don't have access um if you're on Google collab and if you don't have access to the a100 that's completely
26:17
fine you can just go to the runtime and you can just switch to a T4 T4 GPU which
26:22
is free uh and the code which I'm sharing will still run so these models are pre- Trend I'm just loading these
26:28
models from hugging face and then I'm using this prompt the quick brown fox
26:34
jumps over the lazy dog okay I'm using this prompt and so first in the step one
26:39
what I'm going to do I'm just loading these two models so here you'll see the loading has completed so this took
26:44
around 3 to 4 minutes for me on a100 on a T4 GPU it might take 10 to 15 minutes
26:50
and overall you will need to allocate around uh I think throughout this full exercise
26:56
you will need to allocate around 5 GB of space that's it then what I'm doing is that I'm actually running uh I'm passing
27:04
this prompt through this entire model and uh I'm evaluating the time it
27:10
takes um for the model to give me the attention scource essentially so I'm evaluating the time it takes for my
27:17
input P prompt to pass through the entire model now remember
27:22
that uh multi-ad attention we are using separate key values per head and M mul
27:28
query attention we are sharing the key values across all the heads and multi- head attention the memory consumption is
27:35
higher multiquery attention the memory consumption is lower why is the memory consumption lower in multiquery
27:41
attention because we reduce the number of heads so remember the KV cache size formula for multi-head attention versus
27:48
multiquery attention for multi-head attention is this for multiquery attention we reduce uh by 1 by n so we
27:56
we get rid of this number of heads so for multiquery attention we
28:02
expect to reduce the memory consumption so overall I would expect a good performance time in my multiquery
28:08
attention because I'm saving less amount in the memory and I expect that to improve my performance and I'm also
28:15
doing less number of computations actually because I'm sharing my same keys and values across different heads
28:22
so if you run this second code to compare the inference times you'll see that the inference time for multihead
28:29
attention is 1.62 seconds and the inference time for multiquery attention is 64 seconds so that's I think almost a
28:37
factor of 40% Improvement in the inference time why is there a Improvement in the inference time for
28:44
multiquery attention because we saving Less in the memory uh since it's uh we
28:49
reduce the number of heads in the KV cache formula and we are also doing less number of computations since the keys
28:55
and values are just repeated across multiple I not going through this code because my
29:01
main intention is not to walk you through this code but to essentially show you how these models function and
29:08
what we can diagnose from this already pre-trained models the next thing which we are going to do after this point is
29:15
that we are going to evaluate the uh we are going to evaluate the trainable key
29:21
matrices and the trainable value matrices for these uh models so let me go to the top here and let me show you
29:28
what we are exactly going to visualize uh we are going to visualize
29:33
this so what you'll see is that so I'm going
29:40
to visualize this is WK right and this is WV so for the multi-ad attention we are going to first see Matrix for head
29:47
one we are going to visualize the Matrix for head 2 we are going to visualize the Matrix for head three we are going to
29:52
visualize the Matrix for head four each of these matrices would look different from each other because in the multi
29:58
head attention we don't share these values and we are going to visualize the same thing now in multiquery attention
30:05
also in our code we are going to visualize this head one this head two
30:10
this head three and head number four so in the visualization you will see that
30:15
so let's let me show you directly the visualization again I won't take you through the code so this is the gpt2
30:21
medium visualization so let's first see for head number zero this is the trainable Matrix for head number0 Z it's
30:29
the same thing as the first uh let me go over here it's the same thing as this
30:35
first first head trainable weight Matrix but let's see the dimensions here the
30:41
number of columns here are equal to 64 why so if you see 0 2 4 it goes up till
30:47
64 the number of columns are 64 because the head Dimension is 64 here the head Dimension was two but now this is just
30:53
64 in our example and the number of rows are 15 so the number of rows here are
30:59
actually equal to the number of embedding Dimension and the embedding Dimension is I think 1024 so we cannot
31:06
show all the one24 here so I have just shown 50 rows over here so that's why it's 50 rows and 64
31:12
columns 64 is the head Dimension and 50 is just a choice I made because I can't show the entire one24 rows so take a
31:20
look at this heat map so every color here represents one value this is the heat map for head number one heat map
31:27
for head number two you sorry head number zero and head number one you'll see that the heat map differs from head
31:32
zero and head one right it differs from head one to head two so you'll see that all the keys have different heat Maps
31:39
over here which is expected since the keys don't share the same values even the value matrices so now I'm plotting
31:45
the value Matrix for head zero value Matrix for head one for head two they they all seem to be very different from
31:52
each other now you'll see for the Falcon model this is the key Matrix which has
31:58
been plotted for head number zero which means that this is the
32:03
trainable key Matrix which is plotted for the first head right now and now if you scroll
32:09
down below you'll see that the values for head zero for head one for head two
32:15
for head three for head four for head five they are all the same because all of these heads now share the same key
32:22
values similarly if you go to the value matrices you'll see that for head number zero for head number one for head number
32:29
two for head number three for head number four and for head number five all of these heads share the same Val value
32:37
matrices why do they share the same value matrices because that's the main thing which is implemented in multiquery
32:42
attention the same weight matrices are shared across different heads that's the key advantage of uh multiquery attention
32:51
so I'm just visualizing it here for you this is the second visualization I wanted to show for gpt2 the values are
32:58
different across different heads and for Falcon model the values are same across the different
33:04
heads the last visualization is I want to show you the attention score matrices
33:10
so first you can see for multi-ad attention I'm just looking at the last Transformer block remember there are
33:16
multiple Transformer blocks um and if you see in each of these I'm showing
33:22
that so now if you see I have the quick brown fox the quick brown fox jumps over
33:28
the lazy dog right so there are nine tokens so the attention score Matrix will be 9 by9 let me write it on the
33:35
white board over here the quick brown fox jumps over the
33:42
lazy dog the quick brown fox jumps over the lazy dog so this is the attention
33:49
score Matrix for the first head uh similarly there will be an attention score Matrix
33:56
for the second head Etc I have just shown these values over here I have just shown these values over here for the
34:02
different uh for the different heads you see that's what this heat map actually
34:07
indicates so gpt2 has 16 heads I think gpt2 has 16 heads and you'll see this
34:13
attention score Heat Map for all the 16 heads you'll see that it's different from each other and then for Falcon I've
34:20
shown the same thing but Falcon model has actually 32 heads so for each head I
34:25
have shown the attention score Heat map which is 9 by9 because there are nine tokens and here what I actually want to
34:33
point out is that the main advantage of multi-head attention is that each head can capture a different perspective and
34:39
each head does capture a different perspective so ideally you should see more diversity in what all is captured
34:45
through these different attention heads whereas if you see the Falcon model right now it looks kind of similar but
34:51
overall the effect is that the diversity captured by each head is not that high because the keys and the values among
34:58
these different heads are shared so that's what I wanted to show through these plots that for the Falcon models
35:05
if you see for the different heads the diversity for many of these heads is not that high they look almost kind of
35:10
similar this effect is not that pronounced in this visualization right now but this highlights one mean
35:16
disadvantage of multiquery attention and the disadvantage is that the diversity
35:23
among different attention HS is not that much so we don't capture as many perspectives so our model becomes weaker
35:29
our model does not perform as well all right so this brings us to the
35:35
end of this lecture on multiquery attention multiquery attention is the first method which people or researchers
35:41
invented to solve the KV cach memory problem and the idea is that we share
35:46
the same key matrices and value matrices across all the heads similarly we share
35:53
the same key weight matrices and value weight matrices across all the heads and
35:58
that reduces the KV cache by a size of n so for gpt3 if the KV cach took 4.5 GB
36:06
since we have 96 attention heads for if you use multiquery attention There is
36:11
almost a factor of 100 difference reduction in the amount of memory needed deep seek has 128 attention heads
36:19
so multiquery attention actually reduces the KV cach size by factor of 128 so from 400 GB we go to 3 GB that's the
36:27
good side of multiquery attention the Dark Side of multiquery attention is that it defeats the purpose of
36:33
multi-head attention the main purpose of mha was to capture different perspectives through different attention
36:39
heads in multiquery attention this purpose is defeated because well although different
36:45
attention heads value will be different because the query vectors are different across the different heads but the keys
36:51
and the value vectors are exactly the same so we cannot capture as much diversity so the model per performance
36:57
will degrade the language model will not be as good in capturing the complexity of the underlying sentences or the
37:05
paragraphs in the next lecture we are going to look at another method to solve the KV cach problem and that's the
37:11
grouped query attention we'll also see a code for the group query attention and
37:16
then finally we'll move to the multi-head latent attention which is the key innovation in the Deep seek paper
37:22
thanks a lot everyone I hope you are liking these lectures please make notes now we are getting deeper and deeper
37:28
into the deep seek Innovations so as we get deeper when you follow along things
37:34
might get a bit challenging so make notes you can share the notes with me ask doubts Etc thanks everyone I look
37:41
forward to seeing you in the next lecture
Multi-Query Att




