#### Multi-Token Prediction Introduction

| Year | Paper |
|---|---|
| Apr 2024 | [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737) |


***

* 5:00

***

* 10:00

#### Why is MTP useful?
1. Densification of Training Signals
* MTP provides richer and denser training signals than single token prediction. 
* Traditional single token prediction only guides the model to predict a single immediate token.
* MTP, however, instructs the model to simultaneously predict multiple future tokens, generating more informative gradient signals per training example.
2. Improved Data Efficiency
3. Better Planning
4. Higher Inference Speed

***

* 15:00

* MTP train models achieved better results on standard benchmarks like HumanEval and MBPP with the same amount of training data, solving about 15% more code problems on average.
* [Mostly Basic Python Problems Dataset (MBPP)](https://github.com/google-research/google-research/blob/master/mbpp/README.md)
* [Evaluating Large Language Models Trained on Code (Jul 2021)](https://arxiv.org/abs/2107.03374)
* [HumanEval](https://github.com/openai/human-eval)

***

* 20:00


  
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











