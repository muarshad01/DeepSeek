
***

* 5:00

* __Step-0__: Load Packages
* __Step-1__: Define RMSNorm class
* __Step-2__: Define the Multi-token Prediction (MTP) class
* __Step-3__: Pass input tokens through the model and generate multiple next tokens
* __Step-4__: Calculate loss between target tokens and predicted tokens

***

* 25:00

three prediction tokens as the output for five uh input
26:01
tokens. And once you have understood this, it will be very easy for you to understand the loss function calculation
26:07
which we'll see later also. Now let's come to the next part where we'll actually pass these input tokens through
26:12
the model and generate multiple tokens. So once we have received these output tokens, what we can do or once we have
26:20
received this output tensor, what we can do is we can test our uh MTP class
26:26
implementation to see whether it's working correctly or not. So what I'm going to do is that I'm going to define
26:32
multiple things. So I'm going to define a batch size which is equal to one. I'm going to define the sequence length
26:38
which is equal to 8. So I'm currently matching exactly the input sequence which we have over here. I'm going to
26:44
define my sequence length t is equal to 8 which means the number of tokens here. Then I'm going to define d is equal to 8
26:52
which is the dimension. So I've defined t is equal to 8. d model is equal to 8
26:57
and the vocabulary size I'm defining to be equal to 5,000. Okay. So the first
27:03
thing I'm going to do is I'm going to create an instance of this model using all of these variables which I've just
27:08
defined and then I'm going to create a batch of input tokens which I'm going to pass through this
27:15
model. So tokens dot randt which means I'm going to randomly sample tokens from
27:22
my vocabulary and they will be batch size, the sequence length. So if my sequence length is equal to 8, I'm just
27:29
going to s and if my batch size is equal to one, I'm just going to sample eight tokens. So in this example or in this
27:35
step, what we are doing is we are sampling eight tokens and we are passing these eight tokens through the model or
27:41
through the yeah through the simple MTP model and let's check the logits.shape.
27:47
So first if you run this, okay, simple MTP not defined. It seems that I have
27:52
not defined this class. So let me run this first. And now if I run this again, the first thing I want you to inspect is
27:59
this logits.shape. And logits do.shape is one which is the batch size, 5a 3a 5,000.
28:06
And now you should be able to understand what this 5 3 and 5,000 mean. What is five? Five is basically t minus b which
28:14
means I'm only doing the predictions for this, this, this, this, and this. So there are five input sequences. I cannot
28:20
do the prediction here because at prediction depth three nothing will exist. Okay. So, I'm doing the
28:26
prediction for only five input tokens. That's why we have the five here. And what is this three? Three is because I'm
28:32
predicting three future tokens for each input index. That's why we have the three over here. And 5,000 is because
28:39
that's the vocabulary size. So, that will help me in prediction of my next token. I'm going to look at
28:45
the index which has the maximum probability out of the 5,000 tokens. So,
28:50
that's my logic shape. And I'm just going to print out some other things to inspect. So here, what if I want to
28:56
inspect head uh k=0 at i=0, which means that I want to predict
29:04
the logits which are predicted for i=0, which means i equal to0 and for this
29:11
head for the first head. So that means I should just get a 1a 5,000 vector because I'm now looking at the first
29:17
input token and I'm looking at the first head right now. So similarly you can just do logits
29:24
comma 0 comma 0 because that's k=0 at i=0 and that will give you a logits of
29:31
1a 5,000 shape. Okay. Um you can do several other
29:36
things. So for example if you want to predict the tokens for i equal to0 but for all the different heads if you want
29:43
to predict the maximum token index what you will do is that the logits. So each of these heads have logits right which
29:50
are 1 comma vocabulary size 1 comma vocabulary size and 1 comma vocabulary
29:55
size. You'll just take the index which corresponds to the maximum probability in this that will be the next token
30:01
which is predicted for this head. For the second head among this you'll take the index with the maximum probability
30:08






***


for the third head in this vector you'll take the index with the maximum probability. So basically that will give
30:14
you for I equal to0 what are the next tokens predicted at different heads head 1 head 2 and head three. So to plot this
30:21
you can just do logit 0 arg max dimension equal to minus1 which is that
30:27
index which has the maximum value. So you'll get the three tokens predicted for i equal to0. So remember
30:34
for every input index we'll have three tokens which are predicted right. we just have to query the logit vector and
30:40
look at the index with the maximum probability. So this the reason I'm showing all these print statements is
30:47
that this output tensor which you have obtained this four dimension is is actually very rich. It's very rich in
30:53
the sense that it actually contains information of the next three tokens which are predicted for five input
31:00
sequences. This is very different than single token prediction. In single token prediction, we only instead of this
31:05
three, there is only one here, right? Because we only predict the next token for every input uh
31:12
token. All right. And then the last step which we have to implement in our code is that we have to compute the loss
31:18
function between the target tokens and the predicted tokens. So this is also very important to note. Remember we have
31:24
this 5 BA 5A 3A vocabulary size.
31:30
Now let me show you how you can use this to actually compute the loss function. So the key thing to note here is that
31:37
for every in input token. So first we look for i=0, i= 1, i= 2, i = 3 and for
31:44
i = 4. For i=0 we have three predicted tokens, right? So predicted token 1,
31:51
predicted token 2, predicted token 3. And we have three target tokens. What
31:57
are the target tokens? The target tokens are the actual values actual input embedding values at i= 1, i= 2 and i =
32:05
3. So you have the three predicted tokens and you have the three target tokens. Target token 1, target token 2
32:12
and target token 3. What you do is that you just take the cross entropy loss between these three and add them up
32:18
together. That's the loss for i equal to0. Similarly for I equal to 1 you have
32:23
the predicted token two, predicted token three and predicted token four and you have the target token two, target token
32:30
3 and target token 4. You take the categorical cross entropy loss between
32:35
all these and add it together. So this will be L2, this will be L1. Similarly for I equal to 2 there will be loss of
32:41
L3. I is equal to 3 there will be loss of L4 and I equal to 4 there will be loss of L5. And then what you can do is
32:48
that you can just take the mean of all of these losses L1 + L2 plus L3 + L4 +
32:55
L5 and that will give you the total loss for this input sequence. Similarly, if you have multiple batches, you can just
33:03
average over all of these batches. So that's how the loss is calculated in multi-token prediction tasks. So here
33:10
what I've done is that I've done the same thing. We have looped over i =0 1 2 3 4 like I did I
33:16
= i =0 1 2 3 and 4. And for each I for
33:21
each I the second loop is for the prediction depth. So for each I we are looking at three prediction depth one um
33:29
let me write it with a different ink. We are looking at prediction depth one, prediction depth two and prediction depth three. Right? So for each I I'm
33:37
looking at three prediction depth and for each prediction depth I'm taking the categorical cross entropy loss. So for
33:42
each prediction depth for prediction depth one I take the categorical cross entropy loss. For the prediction depth
33:48
two I take the cross entropy loss. For the prediction depth three I take the cross entropy loss and I add them all
33:54
together. So that's the loss for every I and for every prediction depth. Right?
34:00
So then I add all of the losses together and then I divide by L into D because I'm doing L into D loss function
34:06
calculation. So I just divide it which takes the mean essentially. Uh so if there are five
34:13
positions if you see there are five positions right now right 0 1 2 3 4 5
34:18
and for each I'm doing three losses. So it's 5 into three. So ultimately when I add all the loss terms together I'll
34:25
divide by the by 15 to get the mean loss. So if you can run this right now you'll get that uh the loss function is
34:33
12.133 and then similar to how we did back propagation we'll simply do back
34:38
propagation and then try to minimize this loss and optimize the parameters.
34:43
So the main reason why I showed you this code is for you to get a visual sense of how the forward pass might be
34:49
implemented in the case of multi-token prediction and how we get the loss and how this loss can be implemented during
34:55
the back propagation and I hope now all of this code is understandable to you because when you look directly at the
35:02
multi-token prediction implementation and the loss function calculation which deepseek have shown it's very difficult
35:09
to understand what exactly is going on. Even this figure is quite challenging to understand. So I hope that the
35:15
yesterday's hands-on or the previous hands-on lecture which we had on deep se and today's lecture which supplemented
35:22
that lecture with coding that gives you a much detailed understanding of multi token prediction. The reason I'm sharing
35:28







***



this code file with you is so that you can experiment with this further and that can lead to further avenues of
35:34
research. So until this point now we have covered multi-token prediction in detail which is the one of the three
35:41
fundamental building blocks of deepseek along with the multi head latent attention and mixture of experts. So
35:48
these three formed the key elements in the deepseek architectural innovation.
35:55
Of course, they had innovation in their modeling such as uh reinforcement
36:00
learning, supervised fine-tuning, but it's also quite important to pay a very close attention to their architectural
36:07
architectural innovations. So in this 53page paper if you see uh the
36:12
infrastructure or rather the architecture itself that's just just in the 10 that just in 10 pages uh then
36:19
they have a section on infrastructure and then they have a section on post- training pre-training etc. But to
36:27
understand these 10 pages it takes a huge amount of time and effort because these 10 pages have have 25 equations if
36:33
you see and they have a lot of theory behind each of these equations. So I hope these lectures demystify this for
36:40
you. In the next lecture we'll dive deeper into some other concepts of deepseek such as quantization etc. Um so
36:49
I look forward to seeing you during the next lecture. Again I highly encourage you to make notes run these code files
36:55
on your own and explore various changes on your own that will make you better prepared for research. My ultimate aim
37:01
is after watching these lectures you can start working on your own research problems.
37:07
So yeah, this is how multi-token prediction works step by step. This is how Deep Seek implemented multi-token
37:13
prediction and it made Deepseek both faster and more capable and one of and was one of the key innovations
37:20
um was one of the key innovations implemented in their architecture. Thank you so much everyone. I look forward to
37:26
seeing you in the next lecture.










