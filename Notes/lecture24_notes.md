* __Note__: DeepsSeek used MTP gains only during the pre-training process. During inference, DeepSeek just used STP.

***

* 20:00

transformer there is only one transformer block. So, this 1x8 vector which comes into the transformer block
20:15
the output is again 1x8. So the dimensions of this vector are maintained. And now this vector which
20:21
comes out of the transformer block that's the hidden state for the next prediction depth. So if you remember
20:27
what we saw over here the output of the transformer block is the hidden state input for the next head. So for the next
20:35
prediction depth calculation right. So once you get this hidden state for this current depth what
20:41
you do is that as we see over here this hidden state passes through the shared unmbedd matrix. Why is it called shared
20:48
matrix? The reason it's called the shared matrix is because this matrix will be shared across the different
20:54
heads and as a result we don't have to have multiple unmbedding matrix. So this
20:59
is called as unmbedding matrix or a logit's matrix which transports or which converts the vector from the embedding
21:06
dimension to the dimension of vocabulary size. So what we'll do here is that we
21:11
take the uh hidden state vector we pass it through the unmbedd matrix or the
21:16
logits matrix and then we get this logits vector which which is the vocabulary size which is the size equal
21:22
to the vocabulary size and then what we'll do is that similar to how we predict the next token we'll look at the
21:28
index with the highest probability and then we predict the next token for the first input. So that means that for i
21:35
equal to0 we predict the next token and then we'll compare it with the actual next token to get the loss at the first
21:42
prediction depth. So I hope now you have understood all the operations which happen inside
21:48
one head. So we have the merging first then we have the projection then we have
21:54
passing through a transformer block and then finally we have the logits uh calculation which predicts the next
22:00
token. So all the calculations which I showed you right now is for head number one. So
22:06
these calculations are for head number one. And head number two receives the
22:11
hidden state matrix computed by head number one. Head three receives the hidden state vector computed by head
22:17
number two. That's the important thing here. So this link over here right from hidden state one to the head two from
22:22
the hidden state to the to the head three. That's one of the main innovations which deepse had in their
22:27
multi-token prediction architecture. So all the computations which I showed you right now for the head number one all of
22:34
these computations they are exactly the same calculations which happen for head number two and head number three. So the
22:42
computations remain exactly the same. Uh so let me first go through the
22:47
deepsek paper and now show you that whatever they have implemented in equations number 20 uh let's say
22:55
equations number 21 22 and 23 is not quite difficult once you understand
23:00
what's going on. So always keep this this one schematic in mind where what we do is that in every head first we take
23:07
the RMS norm of the hidden state input to that head and the embedding input to
23:12
that head. We merge the RMS norm of the input embedding and the RMS norm of the hidden state vector into that head. We
23:19
merge them together. That's the merging operation. Then we pass it through a projection layer. This H dash serves as
23:27
the input to the transformer. So this TRM whatever is shown here that's the input to the transformer and whatever
23:34
comes out of the transformer that serves as the hidden state input for the next head or for the next uh token prediction
23:42
at the next depth and then what we do is that we just pass the
23:49
um we pass the hidden state at that depth into the output head and then we
23:54
do the next token prediction for that particular depth. Now you'll understand everything what is
24:00
happening in the deepseek paper. Once you have this schematic in mind instead of this schematic deepseek had another
24:07
similar schematic here but I think this is a bit more complicated because it shows a number of input tokens together.
24:13
Instead I think it's much better to just show the journey the journey for one input
24:19
token. Now for every input token you have three prediction tokens right for
24:25
every input token. So we are currently seeing the input token at i =0. For that input token we have three
24:32
predicted tokens correct and then we have to calculate the loss. So we have
24:37
the actual tokens at the three prediction depths. For example at for i equal to0 the actual tokens will be the
24:43
actual input embedding for i= 1, i= 2 and i equal to 3. So then what do we do?
24:49
We just take the categorical cross entropy loss. Then between the predicted token one, the actual token one, the
24:57
predicted token two, actual token two, predicted token three, actual token three, and we add these losses together
25:02
to give us the total loss. So this is how we get the loss from one token.
25:09
Similarly, we get the loss function for all of the tokens in one input sequence. So it's very similar to single token
25:15
prediction, but in single token prediction, for one input token, we just have one loss. But for multi-token
25:21
prediction for one input token we have multiple loss terms which need to be added together. So this is how multi uh
25:28
multi-token prediction works step by step and overall multi-token prediction makes deepse both faster and more
25:35
capable and was one of the key innovations implemented in their architecture. So as such the paper
25:41
itself or the technique itself was not novel. It was already implemented by meta earlier as I mentioned to you. What
25:47
was really novel is that deepseek made sure that there was causality. There was a connection between the different
25:53
tokens which are predicted because the hidden states were carried over and that's what improved I guess
26:00
their multi-token prediction pipeline compared to the original multi-token prediction
26:06
pipeline. So now this figure you'll you should understand quite easily. they
26:11
first we have the first uh chain of the transformer blocks and
26:17
uh the output here if you remember that serves as the hidden state for k equal to 1. So as we have seen over here if
26:25
you see for k equal to 1 the hidden state here is the hidden state which
26:30
comes after the input tokens pass through the shared transformer block. So that's what they have
26:37
mentioned over here. This out this output over here serves as the first
26:42
hidden state and then um this uh this output over here serves as the first
26:48
hidden state over here. So the yeah this arrow which they have serves as the
26:53
first hidden state. Then here the output of the transformer block serves as the hidden state for the next header. So
27:01
this is the same. So these arrows which I've shown in green right now this arrow and this arrow are the
27:07
same green arrows which I've shown over here it's just that the first hidden
27:12
state actually comes from the uh transformer block output the chain of
27:18
the transformer blocks output. So for k equal to 1 for k equal to 1 the hidden
27:23
state comes from the transformer block output. So that's why if you see their figure you'll see that the first thing
27:29
the first block they show as main model uh because the output of the transformer block serves as the hidden state for the
27:36
first multi-token prediction model and then they have multi-token prediction modules after that. Now remember that
27:42
during pre-training deepseek used all of this multi-token prediction models but during their inference they only use the
27:48
main model. So during inference they did not use the multi-token prediction at all. So they did not do speculative
27:55
decoding which we discussed in the previous lecture. During inference they only used the main
28:01
model for the next token prediction. So it was almost like they used multi-token prediction to exploit its densification
28:09
of training signals etc those properties but they did not exploit the faster inference properties of multi-token
28:15
prediction. For inference they stuck to the single token prediction. uh but for
28:20
pre-training they exploited these advantages which we saw densification of training signals improved data
28:26
efficiency and better planning these three advantages they exploited because of their multi-token prediction pipeline
28:33
but for inference as I mentioned they just use the first block over here which they have titled as main model okay so
28:40
this is the entire deepseek multi-token prediction pipeline so in the previous lecture we saw the intuition behind what
28:47
is multi-token prediction and why mult Multi token prediction is actually useful. Uh and my main aim for today's
28:54
lecture was to simplify these equations and this diagram for you. Right? Because if you read this on your own, equations
29:00
21, 22 and 23 might seem like where are these equations coming from? And what do
29:06
do does this diagram actually mean? There are so many things, right? There is this main model here. Why is this shown along with different MTP modules?
29:13
What is this RMS norm? What is this linear projection? Why do we have a transformer block over here?
29:19
uh all of these questions are there because what is this shared over here
29:25
all of this is not explained directly in the paper it's very difficult to understand that but uh that's why I made
29:33
these notes on the whiteboard so that you can exactly understand what is going on in different heads the key thing to
29:39
understand is for every input token there are multiple tokens predicted at different depths and there is causality
29:45
maintained among these different depths which means that information from the first depth the hidden state is carried
29:50
over to the next depth etc which was not present in the earlier multi-token prediction paper which was this where
29:57
they just calculated the output from every head independently so I hope all of you are
30:03
have now understood multi-token prediction in the next lecture what we are going to do is we are going to code
30:09
a multi-token prediction module fully from scratch so I'm going to take this one step further now where if you have
30:15
understood everything on the whiteboard the next step is to code our own multi-token prediction module because I
30:21
want the foundations for all of you to be very strong but at the same time I want to teach all of you the nuts and
30:27
bolts of assembly code from scratch. I have not found the multi-token prediction code uh anywhere yet. So I
30:34
have assembled this code on Google Collab and I'll show that to you in the next lecture. So thanks everyone for
30:40
attending and I look forward to seeing you in the next lecture.







