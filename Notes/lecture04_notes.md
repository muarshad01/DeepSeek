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

* __Context Vector__

* Context vector is an enriched embedding vector. It combines informatin from all other input elements.

***

* 45:00

* $$\\{x^{1}, x^{2}, x^{3}, x^{4}, x^{5}\\}$$

* $$z^{2}= \alpha_{21} \times x^{1} + \alpha_{22} \times x^{2} + \alpha_{23} \times x^{3} +
\alpha_{24} \times x^{4} + \alpha_{25} \times x^{5}$$
* $$z^{2}$$ [context vecot for $$x^{2}$$]

* What's wrong with this approach?

```
The dog chased the ball, but it couldn't catch it.
```

* Simple dot product only measures basic semantic similarity, which isn't sufficient to resolve more nuanced contextual ambiguities.

***

* 50:00

#### Option-2
* Insted of directly using embeddings, we transform each embedding through trained matrices.

***

* 55:00

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


























