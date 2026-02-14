#### The need for attention mechanism

* Here is how the field of Generative AI has evolved

| Model | Year | (Encoder, Decoder) |
|---|---|---|
| RNN | 1980 |
| LSTM | 1997 |
| Attention + RNN | 2014 |
| Attention + Transformer | 2017 | (Encoder, Decoder)| 
| BERT | 2018 | (Encoder, ---)|
| Attention + GPT | 2018 | (--, Decoder)

***

* [ChatGPT](https://chatgpt.com/)
  * I want to learn about AI. Can you help me?

***

* 10:00

* NN can't deal with memory.
* Context

***

* 15:00

* Context bottleneck in RNN is not good for retaining long-range context.

***

* 20:00

* we need to selectively access parts of the input sequence during decoding.
* __Context Window__

***

* 25:00

* while decoding, we can quantify how much  importance (attention) needs to be given to each input token.

* [Neural Machine Translation by Jointly Learning to Align and Translate - 2014](https://arxiv.org/abs/1409.0473)
  * The is "Bahdanau" Attention Mechanism.
  * RNN + Attention Mechanism
* [Attention Is All You Need - 2017](https://arxiv.org/abs/1706.03762)
  * RNN architecture is NOT required for building DNN (they have context problem)
  * Transformer Architecture


***

* 30:00

#### Self Attention
* Mechanism which allows each position in the input sequence to attend to all positions in same sequence.

***

* 35:00

***

* 40:00


40:30
the input embedding Vector contains no information of the neighboring words but now my context Vector consists of
40:37
information of my neighbors also that information is now baked into my input
40:43
embedding so if you have an input embedding if you have the input embedding Vector which is the uniform
40:50
which I talked about right and if you augment this input embedding vectors with context about the neighbors
41:00
we'll see how this augmentation is done but essentially this leads to something which is called as the context
41:08
Vector so the whole goal of the attention mechanism or the self attention mechanism is to convert all
41:15
the input embedding vectors to context vectors so all of these uniforms so we
41:21
saw U these uniforms right all tokens have a 768 dimensional uniform where
41:27
when they come out of the normalization layer and when we go to the multi-ad attention layer what comes in the
41:33
attention layer is an input embedding Vector what comes out of the attention layer is a context Vector so something
41:40
much richer comes out after we exit the attention block and that's why I marked
41:45
it with a different color the reason it's richer is because now it encodes information of other tokens also so it
41:52
retains context so context Vector is an enriched
41:58
embedding Vector it combines information from all the other input
42:04
elements so in self in self attention context vectors play a very crucial role
42:11
their purpose is to create enrich representations of each element in an input sequence by incorporating
42:17
information from all the other elements in that sequence uh so this is again keep this
42:23
thing in mind that input embedding vectors only contain information about that that that word or that token it
42:31
might encode information about the meaning of that word and its position but it has no clue of the neighbors
42:37
context Vector has clue about the neighbors uh because neighbors are so important right when you look at a
42:43
sentence when you look at a paragraph individual tokens don't mean anything it's only how they relate with the
42:50
neighbors that essentially produces the context of that paragraph um and why is this needed in
42:56
llm it's needed to understand the relationship and relevance of words in a sentence to each other actually this is
43:04
that fundamental thing which has made llm so so much better so if you look at
43:09
this advancements in history right Elisa RNN alist El until your attention was
43:15
not there 2014 I believe is a very critical point that was 10 years back when the attention mechanism was
43:21
introduced and then people started thinking that oh instead of looking at words in isolation what if I take a step
43:27
back and try to see how different words essentially relate to each other so then we are exploiting the maximum richness
43:35
from text because just like images are made up of patterns of pixels text or
43:41
paragraphs only make sense if you take a look at all the words together and how
43:47
they relate to each other so now the question is that okay you have an input embedding Vector uh
43:54
let's say for next how do you convert it into a context Vector so you have an
44:00
input embedding Vector how do you go from the input embedding Vector to the context
44:05
vector and I want you to think about this from the first principles pause this video for a moment and think about
44:12
this right you have the input embedding vector and let's say you have these attention scores how will you modify the input
44:19
embedding Vector so that somehow these attention scores are taken into account
44:24
and you have a context vector so you can pause here for a moment first
44:32
you can try to also think about how these attention scores themselves are computed
44:38
um okay so the simplest thing to do is that let's say if you have
44:44
uh uh this Vector right and if you have all the other vectors why don't we take
44:49
a simple dot product so you have the input embedding Vector for next you have the input embedding Vector for the just
44:57
just take a DOT product between these two that will give you Alpha 2 one just take a DOT product between next and next
45:03
that will give you Alpha 22 uh then take a DOT product between next and day that






***


45:08
will give you Alpha 2 three then take a DOT product between next and is that will give you Alpha 24 and next take a
45:16
DOT product between next and bright that will give you Alpha 25 and once you have all these Alphas
45:22
you can simply do alpha 21 * X1 plus Alpha 2 2 * X2 + Alpha 2 3 *
45:31
X3 + Alpha 2 4 * X4 plus Alpha 25 *
45:40
X5 and why do we take a DOT product here essentially the reason you might think
45:46
of a DOT product is that a DOT product essentially encapsulates information about whether vectors are similar or
45:52
closer to each other or not right if you have one vector here V1 and if if you have another Vector here V2 the dot
45:59
product between them will be higher than let's say V1 and V3 so if two vectors are similar their
46:06
dot products will be higher and that's exactly what you wanted to quantify with the attention mechanism you might think
46:12
that I want to quantify whether two vectors are similar right so if next and the are more similar of course they
46:18
should have a higher attention score so this calculation seems to make sense to me I just calculate the dot products and
46:25
then I just scale the different vectors with their dot products and add it up so
46:31
whatever the sum would be that would be the context Vector now for next and similarly I can find context
46:38
vectors for all the other tokens as well what's wrong with this
46:44
approach why will this approach not work or why can't we just take simple dot
46:51
products to find the attention scores again you can pause here for a moment M and try to think of the context
46:59
which we are trying to encode um I want you to think from first
47:05
principles here I'm going to reveal the answer very soon all right so the main answer is
47:11
that let's say you consider this sentence the dog chased the ball but it couldn't catch it right the dog chased
47:18
the ball but it couldn't catch it and let's say I have the input embedding Vector for dog as this input
47:25
embedding Vector for ball as this and input embedding Vector for it as this okay and if it is my query Vector right
47:32
now um how did you decide to compute the attention score between a query vector
47:38
and other vectors you decided to take a DOT product right this is exactly what I'm going to do if it is my query Vector
47:44
to get the attention score between it and dog let me simply take a DOT product between it and dog and if I take a DOT
47:51
product it's 51 if I take a simple dot product between it and ball that is is
47:56
also 51 you see the problem here both the attention scores are completely
48:02
identical but that's not what I wanted when when you say but it couldn't catch
48:07
it it actually means the ball right the dog chased the ball but it couldn't catch it so the second it means the ball
48:16
not the dog so when I'm looking at it I need to pay more attention to ball not
48:21
the dog let let me write this with a
48:27
different ink so that this is clear I need to pay a lot more attention to ball
48:33
when I'm looking at it and not really the dog but that's not that's not what is
48:39
happening here if you take a simple dot product there is no provision for me to encode the information that ball should
48:46
be given more priority than dog both dog and ball should not be having the same
48:52
attention score when we say it this is actually this example is a
48:57
brilliant demonstration of why we should selectively attend to different tokens the dog chased the ball but it couldn't
49:03
catch it the first it is the first dog so if this was the query token it would
49:08
have attended more to dog but now this is the query token so it should attend more to ball I don't want both to have
49:16
the same attention score so simple dot product cannot distinguish between subtle contextual relationships here it
49:23
doesn't consider the context of Chase couldn't catch or linguistic Nuance such as the fact that catch is more likely to
49:30
refer to a moving object which is the ball so the main issue is that simple do
49:36
product only measures semantic similarity but it cannot deal with
49:41
contextual issues and many sentences might have contextual complexity like this right um and I need to encode a
49:49
mechanism so that I can capture these complexities and I don't know what that
49:55
mechanism would be so then we use the trick which researchers have used for a long period of time now if you don't
50:02
know what the underlying relationship between things is you just replace it with a neural network or a bunch of
50:08
trainable weight metries and let back propagation figure it out and that's
50:14
exactly what happened in the field of attention also researchers essentially could not figure out what that mechanism
50:21
can be so and that's where the field of machine learning deviates or deep learning Ates from physics right in
50:28
physics if you were stuck with this problem you would have spent 6 months one year trying to develop a law for the
50:34
underlying mechanism to capture complexities or underlying mechanism to capture the context but in the field of
50:41
deep learning you don't do that you say that I'll replace it with a bunch of matrices and I'll train these
50:48
matrices through back propagation so that's what researchers did right so they
50:54
invented new matrices which are let's say called as the query Matrix and the key Matrix what it means is that instead
51:01
of just looking at the input embedding representations What If I multiply every
51:07
input embedding with a matrix so if my query here is it right my query is it
51:15
I'll multiply it with something which is called as the query Matrix this can be a high dimensional
51:21
Matrix um for dog so dog and ball are the keys right
51:27
uh because keys are essentially if you have the query keys are essentially all the other tokens which you're looking
51:33
for so that's dog and ball so you you multiply both of them with a keys Matrix
51:38
now see the advantage here is that if a DOT product cannot get the contextual relationship you are hoping that these
51:45
WQ and WK you are not assuming these Matrix as anything you are just you will
51:50
initialize them randomly and then you will train them through back propagation
51:56
it's the same deep learning trick which researchers now have used for a very long time if you cannot figure out the
52:02
relationship yourself you take a step back and you let neural network do its job instead of restraining the neural
52:09
network by imposing some laws let it figure it out itself so you see the advantage is now we have multiple uh
52:16
trainable factors in our control so if WQ is let's say
52:21
3x3 and WK is 3x3 right um so dog ball
52:26
and it these are my keys and if these are the embeddings for these which we saw here also the input
52:34
embeddings which we saw and now I multiply these input embeddings with the query I multiply
52:41
this these I multiply these two with the keys so I will multiply so it will be
52:47
3x3 multiplied 3x 1 so this will be a 3x1 and this will also be a
52:55
3x1 so then the keys become .92 and 0.1 And1 1.8 And1 you see these values
53:03
changed because I multiplied them with the keys Matrix and the query is
53:09
it so it will be multiplied with the queries Matrix so that the query for it
53:15
will become 0.5.1 and. 5 1.0
53:21
And1 uh and so now if you plot this in Vector space this is the query Vector
53:27
this is the keys for ball and this is the keys for dog so now we have we are going from the input embedding space to
53:35
a different space which we get after multiplying with the queries and the key Matrix and now I will compute the
53:41
attention scores between these vectors not the original vectors so now if you compute the attention score between it
53:47
and the ball you'll see that it's 56 it and the it and the ball
53:53
is96 and if you compute the attention SC score between it and the dog that is 56
53:59
so here you see the attention score between it and ball is96 Which is higher
54:06
than the attention score between it and dog which is lower so these are clearly distinct
54:12
attention scores so adding these trainable matrices has actually helped
54:17
us why has it helped us because it has given a number of parameters to tune so
54:22
that we can encode some complex relationships between tokens so if you take a simple dot product the attention
54:29
scores will be identical but if you take if you have the query key Matrix we have
54:35
not yet seen the value Matrix we'll see that in the next class but essentially if you just have trainable
54:41
matrices then you can have attention scores which are different because now you suddenly have more parameters to
54:47
work with so if you got confused in this part let me repeat it once more um we started
54:54
this section by thinking that if you have an input embedding Vector right what can you do to the input embedding
55:00
Vector to get the context Vector so to get the context Vector we essentially need Alphas after you get the alphas
55:08
then you just have to multiply them with the um input embedding
55:13
vectors uh and then you will get the context Vector but then the question is that how do you get the alphas between
55:21
one uh input embedding vector and another input embedding Vector how do you get the attention scores the
55:27
simplest way is probably taking a DOT product but we saw that let's say if this is the sentence right and if it is
55:34
my query and if I I want to find the attention score between it this it dog
55:40
and ball I will take a DOT product between it and the ball first which comes out to be0 51 and I will take a
55:47
DOT product between it and the dog which comes out to be again 51 so the attention scores comes out to
55:54
be similar but this is not what I wanted because when I say it I want it to be
55:59
the ball so I want the attention score between it and ball to be much higher than the attention score between it and
56:07
dog so how to do it now dot product clearly does not have the complexity to
56:12
capture these contextual relationships I need more parameters to work with I need
56:18
some knobs which I don't know currently but let neural networks or let back propagation figure out what those knobs
56:24
can be at least let me initialize it randomly for now and that's where this new terminologies coming to the picture
56:31
right I want to have new trainable matrices let me call this query Matrix
56:37
and let me transform the input embeddings into another space by multiplying it with the query Matrix and
56:44
the input embeddings for the keys which are the dog and the ball they'll be multiplied with the keys Matrix and
56:51
we'll transform it into another space and then I will find the attention scores in that transformed vectors
56:56
between those transformed vectors and now if my model learns these parameters of these matrices correctly I can get
57:05
the model to learn that the attention score between it and the ball is96 Which
57:10
is higher than the attention score between it and the dog which is 56 don't worry about these
57:17
multiplications or mathematics right now I I'll do the mathematics in detail in the next lecture for now just remember
57:24
that we we don't know how to physically capture the contextual relationship so
57:30
it's like an easy way out it's a trick you you introduce the queries you introduce the keys randomly these are
57:36
random trainable matrices you initialize them randomly and then you train them so
57:42
you might have heard of this word query Keys there's actually no proper physical reason why they are introduced the only
57:49
reason they are introduced is because humans could not figure out how to capture these attention scores
57:54
themselves the only way we know is that okay if we cannot figure it out let me project my input embeddings into higher
58:02
Dimensions or different dimensions or let me have few trainable parameters to work with and then
58:08
hopefully the training itself will figure it out on its own and this trick humans have done in
58:14
the field of computer vision also if you train a CNN to distinguish between dogs and cats you cannot write down all the
58:22
features yourself you rely on a convolutional neural network to do that it's kind of a similar thing over here
58:30
in the next lecture we are actually going to see how do we exactly compute the queries Matrix the keys Matrix and
58:37
there is also one more Matrix which is called the values Matrix how that is used in the next token prediction task
58:45
that we are going to see in the next lecture so the next lecture is all about the next lecture is about the
58:52
mathematics of self attention mechanism what do we do with the queries Matrix key Matrix and the values Matrix exactly
59:00
how do we calculate the context vectors mathematically and from those context vectors
59:06
ultimately uh what do we do in all these steps to get the next token prediction
59:13
so next class is pretty much going to be a deep dive into this section which we
59:18
just saw uh and expanding it into a full lecture of mathematics but now in
59:23
today's lecture I just want to motivate this concept of queries keys and values values we have not seen yet we'll see
59:30
that in the next lecture all right everyone so this brings us to the end of today's lecture
59:36
which you can think of as a mixture of the history of the attention mechanism plus an introduction to self attention
59:43
for the next token prediction task as a summary remember how the attention mechanism has Evolved first we had uh
59:51
Elisa uh Elisa was a revolution at that time and it's pretty awesome considering it inv got invented in
59:58
1966 then came recurrent neural networks and lsms they had the context bottl niic
1:00:04
issue which means that all the context was compressed into just one hidden state to solve that we understood that
1:00:10
we needed to selectively pay attention to different parts of the input sequence and that is what is called as attention
1:00:17
so to encode that we introduced something called as the attention mechanism which computes the attention
1:00:22
scores between the decoded output or the decorder hidden States and the input hidden states that paper was the badana
1:00:30
attention mechanism published in 2014 uh that paper essentially still
1:00:35
had RNN so that was attention plus RNN in 2017 there came a paper in which
1:00:42
researchers realized that we don't even need rnns so they scrapped out rnns and
1:00:47
they came up with a new architecture called the Transformer architecture which had the attention mechanism at the
1:00:52
heart of it 2018 researchers modified the Transformer architecture they
1:00:57
scrapped the encoder kept the decoder and had this architecture in which the
1:01:03
attention mechanism was again at the heart of it uh so this until now the attention
1:01:09
mechanism was from one sequence to another sequence then when we talk about self attention we essentially look at
1:01:16
just one sequence because that will be used for next token prediction tasks so
1:01:22
in next token prediction tasks like GPT we use self attention where we look at one token and how it
1:01:28
attends to its surrounding or neighboring tokens so the token which we are looking
1:01:33
at is called as query and the other tokens are called as keys and we want to find the attention score between the
1:01:39
query vector and the keys we realize that the main purpose of the attention mechanism is to get these attention
1:01:46
scores and to convert it into context Vector context Vector is a more enriched
1:01:51
version of the input embedding Vector because it also contains information about how one token relates to its
1:01:58
neighbors to get these attention scores the naive way or the simplest way to think about it is just to take a DOT
1:02:05
product between vectors but we realize that that's not the best way to go about it because just taking a simple dot
1:02:12
product can't capture subtle contextual relationships like we saw in this
1:02:17
example the dog chased the ball but it couldn't catch it the first it is the dog the second it is the ball to capture
1:02:24
such contextual complexities we need to add trainable weight matrices so we need to increase
1:02:32
the parameters so that we have different knobs to play around with these trainable matrices are called as the
1:02:38
query weight Matrix and the key weight Matrix there is also value weight Matrix which we'll see in the next class the
1:02:44
input embedding for the all input embeddings are multiplied with the query weight Matrix to get the query Matrix
1:02:51
and also we have the keys Matrix like that so then the attention scores are not found between the input embeddings
1:02:57
of the vector they are found between the queries and the keys and since we have a flexibility of
1:03:04
so many parameters to play with we hope that when we train the parameters they will learn that the attention score
1:03:10
between the it the second it second it and the ball is higher than
1:03:18
the attention score between the second it and the dog so it captures more
1:03:24
contextual complexities so addition of these trainable weight mates captures more contextual complexities and that's
1:03:32
why we humans added these weight matrices and then we call them queries
1:03:37
keys and values because it it sounds cool and it also it relates to the field of information Theory but if you look at
1:03:44
it deeply we cannot figure out the rule for this attention mechanism ourselves
1:03:50
do product fails so we cannot figure out how to get these attention scores how to compute them ourselves so we turn to
1:03:56
neural networks to do the job for us um all right so thanks a lot everyone
1:04:03
in the next lecture we'll be diving deep into the mathematics behind self attention in the lecture after that
1:04:09
we'll look at multi-head attention and only then we'll be so this is the multihead attention notes and only then
1:04:16
we'll be truly ready to understand uh the key value
1:04:22
cache so let me see where that is yeah only then will be truly ready to
1:04:27
understand the key value cach which serves as the segue for the multi-head latent attention which is
1:04:33
MLA this series is going to be a bit deep but I'm trying to make the lectures as long as possible so that I don't miss
1:04:41
out anything this is for serious Learners so please make notes as you are watching this series and it will be
1:04:47
incredibly useful for you thanks a lot everyone and I look forward to seeing you in the next lecture




















