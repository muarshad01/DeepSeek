#### Multi-Token Prediction Introduction

* [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)

veryone, my name is Dr. Raj Dandkar. I graduated with a PhD in machine learning from MIT in 2022 and
0:08
I'm the creator of the build deepseek from scratch series. Before we get started, I want to introduce all of you
0:15
to our sponsor and our partner for this series, Invido AI. All of you know how
0:20
much we value foundational content building AI models from the nuts and bolts. Nvidia AI follows a very similar
0:28
principle and philosophy to that of us. Let me show you how. So here's the
0:33
website of Invido AI. With a small engineering team, they have built an
0:38
incredible product in which you can create highquality AI videos from just
0:43
text prompts. So as you can see here, I've mentioned a text prompt. Create a
0:49
hyper realistic video commercial of a premium luxury watch and make it cinematic. With that I click on generate
0:56
a video. Within some time I'm presented with this incredible video which is
1:02
highly realistic. What fascinates me about this video is its attention to detail. Look
1:08
at this. The quality and the texture is just incredible. And all of this has been created from a single text
1:15
prompt. That's the power of Invido's product. The backbone behind the awesome
1:21
video which you just saw is Invido AI's video creation pipeline in which they
1:26
are rethinking video generation and editing from the first principles to experiment and tinker with foundational
1:33
models. They have one of the largest clusters of H100s and H200s in India and
1:38
are also experimenting with B200s. Nvidia AI is the fastest growing
1:44
AI startup in India building for the world and that's why I resonate with them. so much. The good news is that
1:51
they have multiple job openings at the moment. You can join their amazing team. I'm posting more details in the
1:57
description [Music] below. Hello everyone and welcome to
2:05
this lecture in the build deepseek from scratch series. Today we are going to
2:11
get started with a very important module and that is called as multi-token prediction.
2:18
When you look at the deepseek architecture, they had three predominant revolutions in their architecture. The
2:26
first one was multi head latent attention. The second was the innovations which they did on top of
2:32
their mixture of experts module and the third one was a very smart implementation of something which is
2:39
called as multi-token prediction. Today we are going to learn about this third technique. In the
2:45
previous lectures we have covered everything which deepseek implemented related to multi head latent attention
2:51
and mixture of experts. So if you go to the paper uh which
2:56
deepseek released in January 2025 and which led to this whole deepseek
3:02
revolution that's deep three deepseek v3 technical report and if you scroll down
3:08
below you'll see that first they have this u in their
3:14
architecture section first they have the multi head latent attention schematic then after this there's a
3:21
whole section on the mixture of experts module which they have with innovations such as auxiliary loss free load
3:27
balancing shared experts fine grained expert segmentation etc. And finally the
3:33
last thing which they have is this multi-token prediction and this is what we are going to see today. This is the
3:40
schematic which deepseek has with respect to their multi-token prediction module. Now this looks simple that okay
3:48
we usually do single token prediction in language models. What is this multi-token prediction? Maybe we are
3:54
predicting multiple tokens. But there are actually a lot of complexities which which are involved in this process. And
4:01
that's why we are going to take about two to three lectures for me to explain this entire concept of multi-toal
4:09
prediction to you in a lot of detail. As always, we are going to uh show you
4:14
everything on a white code and then I'll also take you through the code of how can we code a multi-token prediction
4:20
module from scratch. So that's the plan for the next two to three lectures. So first let me
4:27
take you through the history of this multi-token prediction. Actually multi-token prediction was not invented
4:34
by deepseek. Deepseek just built on top of it like they have done with so many of their other architectural
4:40
innovations. Multi-token prediction was first implemented in this paper called better and faster large language models
4:48
via multi-token prediction and this paper was uh a group of researchers
4:55
contributed to this paper and one such group came from meta
5:00
uh deepse built upon this paper. So this paper was published in April of 2024 and
5:05
DeepSync immediately took this they built on top of this and implemented that implemented that in their version 3
5:13
architecture. Okay. So if you look at the abstract this these authors say that large language models are trained with
5:20
next token prediction loss. In this work we suggest that training language models to predict multiple future tokens at
5:27





***

once results in higher sample efficiency. So that's what multi-token prediction essentially is. We are going
5:33
to predict multiple future tokens at one time. So let us start understanding the
5:40
difference between single token prediction and multi-token prediction. In this lecture, I'm just going to
5:45
motivate the concept of multi-token prediction. I'm going to show you how it differs from single token prediction and
5:52
then we are also going to discuss some advantages of multi-token prediction. So I'm going to build your intuition in
5:58
this lecture. In the next lecture we are going to see how deepsek exactly implemented the multi-token prediction
6:05
architecture and in the lecture after that we'll implement multi-token prediction in
6:11
code. So let's get started with today's lecture. If you take a look at single token prediction, the pipeline for
6:18
single token prediction looks something like this. Let's say we have a bunch of input tokens which look something like
6:25
this. uh and let me actually let me take a batch of eight input tokens. Okay.
6:34
Artificial intelligence is changing
6:41
the world right now. Let's say I have this batch of
6:46
eight input tokens. The way single token prediction works is that this entire batch passes through multiple sequences
6:54
of the transformer blocks. So here I have shown three transformer blocks. Ideally there could be 12, 24, even 36
7:01
transformer blocks and these are called as the shared transformer trunk. That's
7:06
the terminology introduced in this first multi-token prediction paper. So I'm
7:11
going to utilize this same terminology. What this essentially means is just a bunch of transformer blocks chained
7:18
together, right? And then finally, so let's say I have this bunch of transformer blocks D1, T2, etc. D3,
7:25
right? Up till D12 which is chained together and after that I have my logits
7:31
matrix which converts from which converts every token from the embedding dimension to a vocabulary size. So now
7:38
if you have this uh let me copy paste this these eight tokens when they come out uh of the
7:45
transformer blocks the final output is the logit's token where or the logits matrix where every token has now
7:53
if the vocabulary size is 50,000 every token has now a 50,000 dimensional
7:59
vector corresponding to it. So all of these tokens have the 50,000 dimensional
8:05
vector corresponding to it. And then we look at that token index which has the
8:11
highest probability associated with it. And then we get the next token which is predicted. Next token which is predicted
8:18
for all of my input tokens. And during inference time the only next token which
8:23
matters is the last token. That's the new token which is inferred. But during training time all of these tokens which
8:29
are predicted they will be used for getting my training loss. So for example for the first token which is predicted
8:37
t1 the actual prediction should be intelligence right. If artificial is the
8:42
input the actual output should be intelligence over
8:48
here. For the second input which is artificial intelligence the actual output should be so this is actual
8:55
output. The actual output should be is here the actual output should be changing etc. So we have the actual
9:02
output and then we have the predicted output and during training the loss is computed between the actual output and
9:08
the predicted output. That's essentially the single token prediction task. In multi-token prediction task
9:16
what changes is that for every token which I'm looking at. So let's say if
9:21
I'm looking at artificial here in single token prediction only one token is predicted right in multi-token
9:27
prediction instead of one token there are three tokens which are predicted let's say let me call it s_ub_1 s_ub_2
9:33
and s3 these are the three tokens which are predicted during multi-token prediction for every input token and for
9:40
these three I have my actual I have my actual tokens which are s1 dash s2 dash and s3 dash and then I
9:48
get the loss between my predicted three tokens and the actual three tokens for every input token. So if you look at
9:54
artificial right now, you first define the horizon into the future. And let's
10:00
say I want to look at three tokens into the future. So the actual three tokens are intelligence, is and changing. So
10:07
S1, S2 dash and S3 dash are intelligence, is and changing. And the predicted three tokens might be
10:13
something completely different. S1, S2 and S3. And then you take the loss the same cross entropy loss between S1 S2 S3
10:20
and S1- S2- S3 dash like you did previously. It's just that the future
10:26
horizon which I'm looking at for every token now changes. So earlier when we looked at each token when I looked at
10:32



***



artificial the only future horizon which matters is one token right which is intelligence. But now when I look at
10:38
artificial I'm going to take into account the next three tokens which also means that for every input token we are
10:46
actually predicting three tokens into the future. Right? So whenever you are doing inference instead of predicting
10:52
one token now we are predicting three tokens into the future. So multi-token prediction also has
10:59
profound implications on what happens during inference because now we are not just predicting one token but we are
11:05
predicting three tokens into the future. Okay. Uh so this is the main uh
11:12
conceptual difference between single token prediction and multi-token prediction. And the way I have shown it
11:18
in the schematic here is that if you take a look at single token prediction, you have a shared transformer trunk and
11:24
only one single token is predicted at a time. Whereas if you look at multi-token prediction, the architecture is somewhat
11:32
different and we'll see that in the next lecture. But multiple tokens are predicted for every given input token.
11:38
That's something which is very important for all of you to keep in mind. Now you might be thinking why are
11:44
we really doing this? Because it seems that single token prediction also gives me one token at a time, right? Why do I
11:52
need to predict multiple such tokens at a time like I told you right now? And why do we need to find the loss between
11:58
the actual multiple tokens and the predicted multiple tokens? What is the real advantage of multi-token prediction
12:05
and why is it really useful? So now we are going to see the intuition behind what makes multi-token prediction use
12:11
multi-token prediction useful and for that we are going to refer a lot to the paper better and faster large language
12:18
models via multi-token prediction. So let's get started. So actually when you
12:23
look at multi-token prediction it seems that we are making a simple change over here. We are just predicting multiple
12:30
tokens instead of one. But it has profound implication on several things.
12:35
In particular, there are four major reasons why multi-token prediction is useful. The first reason I have bucketed
12:42
it into this title called densification of training signals. The second reason I
12:47
have bucketed it into a title called improved data efficiency. The third reason is better planning and the fourth
12:54
reason is higher inference speed. I'm going to now walk you through all of these reasons step by step.
13:02
Um and this paper which Meta and other researchers released in April of 2024,
13:10
they actually had a qualitative justifications for each of the reasons which I'm showing you right now. So
13:17
whatever I'm explaining to you right now, it's not just something which is my thinking or which is my point of view.
13:24
It's something which has been backed quantitatively. And at several points in this discussion right now I'm going to
13:30
show you those quantitative results. So the first thing is densification of training signals. So
13:36
what does this really mean? It means that multi-token prediction
13:41
uh provides richer and denser training signals than single token prediction. So
13:46
traditional single token prediction only guides the model to predict a single immediate token. Whereas in multi-token
13:54
prediction, we instruct the model to simultaneously predict multiple future tokens, generating more informative
14:01
gradient signals per training sample. So when you're looking at a training sample, instead of just predicting one
14:07
token into the future, you are now predicting multiple tokens into the future. Right? That means the model is
14:13
becoming a bit more smarter than it was when it was just predicting one single token. And actually I asked GPT 4.5
14:22
regarding uh what does this densification actually really mean. So here is some good series
14:30
of explanations. Um so now in multi token prediction the model actually
14:35
learns about longer range structure grammar and coherence directly from each
14:41
training sample. Why? Because for each training example we are predicting a lot into the future. Right? That's why the
14:48
model learns about longer range structure and grammar. This is why we say that the model becomes rich during
14:55
training when we do a multi-token prediction. So with single token
15:00
prediction, the model only learns immediate next steps dependencies. But with multi-token prediction, the model
15:07
sees and learns the relationship across multiple future steps simultaneously.
15:13
And this guides the internal representations towards better planning or forecasting of sequences. That's very
15:20





***


important. So now as the model is learning, the model is getting a bit better at planning and looking ahead
15:26
into the future. Both of which are very important skills for a language model to behave like humans and for a language
15:33
model to have better performance. And all of these things happen because
15:38
now instead of predicting one single token into the future, we are predicting multiple tokens into the future. Right?
15:45
Uh so that leads to more richer and more informative gradients also. So when you read literature on
15:53
multi-token prediction, you'll often see that multi-token prediction is done so that uh because pre-training is much
16:00
more richer. That is because the model sees and learns relationship across multiple future steps simultaneously and
16:07
it gets better at planning or forecasting of sequences. So that's the first main reason why multi-token
16:15
prediction is useful. It's called densification of training signals or another way to put it is that the model
16:21
just becomes more richer in planning for the future. The second thing is improved data
16:28
efficiency and this is something which the paper which I showed you meta's paper they actually quantified this. So
16:34
they showed that multi-token prediction train models achieved better results on standard benchmarks like human eval and
16:41
MBPP with the same amount of training data solving about 15% more code
16:47
problems on average. So this is just backed by quantitative evidence right.
16:52
So I'm mentioning two frameworks here, human eval and MBPP. First let me show
16:58
you what MBPP is. MBPPP is a benchmark which is mostly basic Python problems
17:03
data set. Uh and as the name suggests the benchmark consists of around thousand
17:09
crowdsourced Python programming problems. And uh the second uh benchmark is called as human
17:17
eval which came out of this paper evaluating large language models trained on code. This paper has become very
17:24
popular now because this benchmark actually uh if you scroll down below on this
17:29
paper here are some questions which are presented in the benchmark. So everything which is marked in the white
17:36
in the paper uh everything which is marked in the white is the prompt which is provided to the model and uh yellow
17:43
so everything which is marked in the yellow uh is the model output right. So this is
17:49
the prompt which is provided to the model everything in the white and yellow is the expected model output. So these
17:55
are some sample questions which are in the human eval. What this paper essentially showed is that if you take a
18:01
look at this plot this is MBPP and human eval and uh you'll see that on the
18:09
y-axis it's essentially the performance and on the x-axis it's essentially the model
18:15
size. So you'll see that for smaller model sizes for small model smaller
18:20
model sizes we are on the negative side right which means the multi-token prediction actually performs poorly but
18:27
as the model size increases so as we go from left to right in every graph here we see that uh the multi-token
18:34
prediction actually performs better right so these bars actually show the performance of multi-token prediction as
18:41
compared to uh single token prediction and we see that for smaller model sizes
18:46
around these model sizes. Uh the bars are negative which means the multi-token prediction performs worse. But as the
18:53
model gets uh bigger and bigger multi-token prediction models outperform
18:58
the baseline when the models become bigger and they significantly show an improvement in both these matrix MBPP
19:06
which is mostly basic uh Python problems and second is the human eval benchmark.
19:13
The second uh plot or the second result which they have is they again took
19:19
several benchmarks like MPPP, human, eval, a/ intro and then here n is the
19:25
number of future tokens which are uh predicted in the multi-token prediction.
19:31
So here also we can consistently see that as we increase the number of tokens. So as we go from top to bottom
19:38
uh as we increase the number of tokens which are predicted the performance on all the benchmarks actually increases
19:43
from top to bottom. So it's 19.3 32.3 42.4 50 etc. So you'll see that uh
19:52
uh as the training data so this is shown for different amount of training data also and in all of these we can see that
19:59
as the number of u u number of tokens which are predicted
20:05
increases the performance increases on that particular benchmark. So here in the NVP benchmark we can see that the
20:12
performance increases from 40.7 to 43.1. Here also it increases from 65.4 4
20:18
to 65.9. Here it increases from 83 to 86.2 and the number of tokens which are
20:24
predicted increase from 1 to 4. Right? So in this second uh factor which
20:30
is called as improved data efficiency, the authors actually proved that multi-token prediction train models
20:37
achieve better results on standard benchmarks like human eval and MBPP. So
20:42
this is something which is quantified now. it just leads to better results especially at coding related problems.
20:49
So these are the first two reasons for why multi-token prediction is useful. But one of the most important reasons
20:56
for uh building the intuition regarding multi token prediction is that they are
21:02
good at planning. So multi-token a multi so this should be mult multi-token
21:08
prediction implicitly assigns greater importance to choice points which are key tokens that significantly influence
21:15
f future outcomes. Thus the model learns to prioritize crucial decision-m elements. So let me go over this once
21:23
more. What this uh what is mentioned here is that multi-tokenal prediction
21:28
implicitly assigns greater importance to choice points. And what is meant by choice points? It's key tokens that
21:35
significantly influence the future outcomes. And as a result, the model learns to prioritize crucial decision-m
21:42
elements and thus they are much better at planning. Let me explain this to you further. So in the paper
21:50
uh they have a section on why does multi-token prediction work and uh it's called some speculation. In this they
21:56
have this figure right and this same figure I have brought here into the whiteboard. So in this figure we have
22:03
model predictions and we have ground truth right. So three tokens are predicted for every ground truth. So if
22:09
the ground truth is one the next three tokens are 2 3 4. The actual next tokens
22:14
is two. When the input is two the next tokens predicted are 3 4 5 but the
22:19
actual next token is three. When the input token is three the next tokens which are predicted by the model are
22:26
four five and a. And the actual token is four. When the actual token is four, the
22:32
next predicted tokens are five, A and B. But the actual next token is five. When
22:37
the input is five, the next three tokens predicted are A, B and C. And the actual token is A. Similarly, when A is the
22:44
actual token or A is the input token, B is the next token, etc. Now, if you look
22:49
at the ground truth, can you identify that point which can be which can be
22:55
classified as a choice point? What is that key token which significantly influences future
23:03
outcomes? So you can pause the video for a moment here and answer this question. What is the key token here in this
23:09
ground truth sequence? In this ground truth sequence which significantly influences or impacts the future
23:16
outcomes. So one gives two, two gives three, four gives five that much is fine and A gives B that is also fine. But the
23:23
key token here which actually can be called as a choice point is this five
23:28
leads to a right because it's at this point where a complete transition happens. We are predicting numbers
23:35
initially sequentially and then suddenly we go from predicting numbers to predicting alphabets. So this is a key
23:42
point or this is a choice point. Okay. Now what I want to mention
23:48
is that uh please keep in mind among all of these predictions there is there are multiple
23:55
places in which token A is actually coming. So three right when three is the input token although the correct answer
24:02
is four for the next token our model also produces a as one of the future tokens when five is the input token our
24:10
model also sorry when four is the input token our model here also produces a and
24:17
when five is the input token of course the model produces a right so what I want to point out here is that token A
24:23
is actually a part of multiple predictions right it appears in the pred prediction starting
24:30
from let me change the color here it appears in the predictions of three four
24:35
as well as five so errors which are related to
24:40
predicting a appear repeatedly in the loss calculation so as we'll see tomorrow when we define the loss
24:48
function the loss function for every input token consists of all of its predicted tokens so the loss function
24:54
now consists a year it consists of a year it consists of a over
24:59
here. So the errors related to predicting a appear repeatedly in the loss calculation.
25:05
Right? Hence the training process implicitly prioritizes improving predictions of such consequential tokens
25:13
focusing the model's capacity on more critical decisions rather than inconsequential ones. Uh now in the
25:21
overall loss of the language model in the multi-token prediction
25:27
uh the prediction of the token A will have a much higher weightage much higher
25:33
implicit weightage because it comes up multiple times. It does not just come up
25:38
when five is the input. This token A actually comes up when three is the input, when four is the input etc. So it
25:46
it plays a big role in the overall loss calculation and here again the
25:53
multi-token prediction loss hence assigns higher implicit weights to consequential tokens. So A is a
25:59
consequential token now right and when we look at the multi-token prediction loss tomorrow we are going to see that
26:06
the multi-token prediction loss uh implicitly assigns a higher weight to
26:12
these consequential tokens and what this means is that now we just get better at planning into the future right
26:18
especially for such choice for such choice points the model learns to prioritize crucial decision making
26:25
elements which are very important so in this particular case uh going
26:31
from uh going from five to A was an important
26:37
um choice point and the model assigns a higher loss to A thereby making sure that we get this transition correct and
26:44
thereby making sure that we are prioritizing this crucial decision making transition from five to
26:51
A. So I'm explaining this at at a bit more intuitive level. No need right now
26:56
to understand it mathematically because we are going to cover this tomorrow. I'm just trying to provide an intuition to
27:02
all of you regarding why multi-token prediction is actually good at planning also. Uh because you'll often hear that
27:09
multi-token prediction tasks are good or multi token prediction models are good at
27:15
planning. So here what they have mentioned is these choice points and uh
27:20
here they have mentioned that multi-token prediction multi-token prediction implicitly assigns weights to
27:26
training tokens depending on how closely they are correlated with their successors. So now this a a is closely
27:34
correlated with it comes in when five is the input and it comes in when three and
27:39
four both are the input right. So there is a higher weightage which is given to the loss which includes a
27:48
um okay so inconsequential transitions following a choice point are hard to
27:54
predict in advance. By marking and counting loss terms we find that n token
28:00
prediction associates a weight of n into n + 1 by2 to choice points and a smaller
28:06
weight to inconsequential points. So don't get confused by this mathematics.
28:11
It just shows that points which are in inconsequential are assigned less weight and points which are more consequential
28:17
such as this transition from 5 to 8 that's assigned more weight. So this is the third reason why multi-token
28:25
prediction is very useful. The first reason is dens densification of training signals. The second reason is improved
28:32
data efficiency. The third reason is better planning. And the fourth reason is with respect to inference. Notice
28:39
that now we are predicting multiple tokens at once, right? And that helps during inference also. It leads to up to
28:45
three times faster inference speed. And this is also mentioned in the
28:51
uh this is also mentioned in the paper. Actually the paper talks a lot about training as well as they talk about
28:57
inference. So if you see section 3.2, it's called faster inference. and they observe a speed up of three times on
29:04
code with an average of 2.5 accepted tokens out of three suggestions on code.
29:09
So it just means that inference actually becomes much faster with multi-token prediction. Now
29:16
whenever you see about multi-token prediction, right? You'll also hear this term which is called as self speculative
29:23
decoding and it comes under the domain of inference, right? So in self-sp
29:29
speculative decoding what is actually done is that if you look on the left hand side during inference only one
29:34
token is predicted at a time but in self speculative decoding we predict multiple
29:39
tokens at a time which means multiple tokens are inferred at a single time through a language model and then
29:46
another larger language model is used to verify those responses. If they are
29:51
correct it those are retained. If they are wrong those are not retained. So this is called as parallel verification
29:57
and this idea is called as speculative decoding which speeds up LLM inference. Now multi-token prediction makes
30:03
speculative decoding possible because in speculative decoding we are considering multiple inferred tokens at once. Right?
30:10
And multi-token prediction is exactly built for this. Through multi-token prediction we can predict multiple new
30:16
tokens uh at a given time. Right? And that's
30:22
why multi-token prediction is often associated with speeding up LLM
30:28
inference and also with self speculative or speculative
30:33
decoding. So as I mentioned speculative decoding involves two models. It
30:39
involves a small language model which quickly predicts the next k tokens in a sequence. Now here is where multi-token
30:44
prediction is used and it also uses a main large language model which verifies the tokens generated by the draft model
30:51
and corrects them as needed. So that's why you'll see the word here self speculative decoding which is used
30:57
during inference. Okay. The reason I'm mentioning all of these terminologies to you and providing a quick intuition is
31:04
because when you hear the term multi-token prediction, you'll often hear the terms densification of training
31:10
signal. then you will hear they're better at planning. Then you will hear the term self speculative decoding. So
31:16
you should have a basic idea of what these terminologies mean. So in summary,
31:22
the four main reasons why multi-toal prediction is useful is that it leads to densification of training signals which
31:28
means that the training just becomes a lot more richer. Um as we saw the key
31:35
thing to note here is that in single token prediction the model only learns immediate immediate next step
31:41
dependencies with multi-token prediction the model sees and learns the relationship across multiple future
31:48
steps simultaneously and uh thus it develops better representations with respect to
31:53
planning and forecasting. That's the first reason which we saw densification of training signals. Second reason is
32:00
improved data efficiency. So this is a quantified result which has been proved
32:05
in this paper where they show that on the MBP and human eval benchmark uh
32:11
multi-token prediction consistently outperforms uh single token prediction.
32:18
The third result or the third reason why multi-token prediction is useful is better planning. And here we saw we saw
32:25
this in a very intuitive manner that multi-token prediction implicitly assigns higher importance to choice
32:31
point. And a choice point is that point which leads to consequential
32:36
uh next tokens. So multi-token prediction loss assigns higher higher implicit weights
32:43
to consequential tokens and it assigns lower implicit weights to inconsequential tokens. That's why it's
32:49
better at planning. And the fourth reason which we saw is it has a higher inference speed up to three times uh
32:56
faster in inference speed. And if you want to implement speculative decoding
33:02
you need two language models, right? You can use a small language model which predicts the next K tokens in a
33:07
sequence. You can use multi-token prediction for this and you can use a larger language model to verify the
33:12
token generated by this small model which is also called as the draft model. Okay. So these are the four main reasons
33:20
why multi-token prediction is actually useful. And now the last thing which I
33:25
want to mention today briefly is that deepseek used multi-token prediction gains only during pre-training. Okay. So
33:34
during inference they did not use speculative decoding. They only used multi-token prediction during
33:39
pre-training because of multiple reasons because they got densification of training signals because of improved
33:45
data efficiency because of improved planning etc. But they did not use the
33:50
higher inference speed of multi-token prediction. In fact during inference they only predicted one single token at
33:56
a time. And this is mentioned in their paper over here. Our MTP strategy multi-token
34:04
prediction strategy only mainly aims to improve the performance of the main model. So during the inference we can
34:10
directly discard the MTP modules and the main model can function independently and normally additionally we can also
34:17
repurpose these MTP modules for speculative decoding. So they have said we can also repurpose this multi-token
34:24
prediction models for speculative decoding. They have not done it but they said that we can do it. But the main
34:29
version three paper used a single token prediction for inference. So during inference they discarded the MTP
34:36
modules. And as they have mentioned at the start an MTP objective densifies the
34:41
training signals. We have already seen what densification of training signal means. And here they also say it may
34:46
improve data efficiency. We have already seen what this this mean. So the reason I have covered this point number one and
34:53
point number two here densification of training signals and improve data efficiency. It's because it's directly
34:59
mentioned in the deepseek paper. Uh here they have also mentioned MTP may enable the model to pre-plan its
35:06
representations for better prediction of future tokens. And we have seen this also here we saw that in the
35:12
densification uh uh the multi-token prediction guides
35:17
internal representations towards better planning. And we also saw that planning is improved in this better planning
35:24
section where we saw that uh the model the multi token prediction implicitly
35:30
assigns greater importance to choice points and thus it's good at decision making into the future. So all of these
35:37
points which have been mentioned in the deepsek first paragraph usually when you read it you feel that you have
35:42
understood it right but beneath it there is a lot of intuition and theory which I wanted to explain in today's lecture
35:50
in in the next lecture on multi-token prediction we'll understand the multi-token prediction modules so
35:57
especially we'll understand the exact mathematics behind how multi-token prediction is implemented by deepseek uh
36:04
so I've written all of this on the white report which we are going to go through sequentially and finally we are going to
36:10
code a multi-token prediction module fully from scratch. All right. So I hope all of you
36:16
found today's lecture useful, informative and intuitive. The main idea behind today's lecture was to give you
36:22
an intuition behind why multi token prediction is used by deepseek and I believe many modern LLMs might start
36:29
using multi token prediction. So I hope you are excited for the next lectures
36:34
which involve MTP or multi-token prediction mathematics and then coding it from scratch. As I'm uh making these
36:41
lectures, it's important that all of you make notes along with me because multi
36:46
multi latent attention and mixture of experts very challenging concepts. Multi-to prediction is not as
36:51
challenging but it's still important that these three fundamental building blocks are very clear in your mind and
36:58
the only way uh these concepts will get stronger and stronger is if you make detailed notes about it. So thanks
37:05
everyone and I look forward to seeing you in the next lecture.


