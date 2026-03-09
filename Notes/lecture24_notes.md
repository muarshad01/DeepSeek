* __Note__: DeepsSeek used MTP gains only during the pre-training process. During inference, DeepSeek just used STP.



Now although multi-token prediction looks simple like it just predicting multiple tokens
1:12
instead of one token. There are a number of different intricacies which we need to understand. For example, if you take
1:20
a look at the Deepseek version 3 paper, you'll see this schematic which they have this schematic for multi-token
1:27
prediction. And there are a number of finer details hidden within this schematic such as this linear
1:32
projection, this RMS norm concatenation, uh, MTP module 1, MTP module 2,
1:39
etc. And when you read this paper, they only have two paragraphs which are
1:44
titled MTP modules over here. And it's quickly explained in terms of mathematical formula. It's actually very
1:51
difficult to understand what's exactly going on here. You feel that you have understood it. But to truly understand
1:57
it, you need to write it down on a piece of paper. So that's what I aim to do today. Today I'm going to show you
2:03
exactly how deepsek implemented multi-token prediction. what these equations mean equation 21 22 23 and how
2:13
to interpret this diagram which deepseek has so let's get started with today's lecture we start out today's lecture
2:21
with what happens in single token prediction right in single token prediction we have a bunch of input
2:26
tokens which go through multiple transformer blocks and then we come out of the transformer block and the input
2:33
tokens maintain their dimension uh so let's say we have three input tokens
2:38
whose dimension is 8. So this is 3x 8. And when we come out of the transformer blocks, we have 3x 8 again that's passed
2:46
through an output head. And let's say if the vocabulary
2:52
size is 50,000. This 3x8 is converted into a 3x 50,000
2:58
matrix. So it might look something like this. Well, this is 50,000. And then
3:04
what we are doing is that for each of these tokens token 1, token two and token three, we look at that index which
3:10
has the maximum probability and that's how we predict the next token for each of the input tokens. So the key thing to
3:17
note here is that for each of these input tokens uh token number one, token number two and token number three, we
3:24
are predicting uh one future token in the single token prediction task through this output head.
3:31
So one output head essentially helps every input token to predict one token.
3:36
So naturally if we want to predict multiple tokens for every input token we
3:42
should have multiple output heads right. So in today's lecture I'm going to
3:47
assume that we are predicting three tokens into the future in multi-token prediction and that terminology is
3:53
called as depth. So the depth which we are looking into the future is going to be equal to three. So we are going to
4:00
need three heads. Head head number one, head number two and head number three. For every input token, head one will
4:07
predict the first future token. Head two will predict the second future token and head three will predict the third future
4:14
token. Now I'm not just going to show you the text with respect to this. I want to show you the mathematical
4:21
intuition behind what I just said. Right? So let's say these are my input tokens. Right? I have eight input tokens
4:27
which go from I=0 I= 1 2 3 4 5 6 and 7.
4:32
So the final input token is I equal to 7 and the dimension of every token is eight. So the number of columns in every
4:39
token is equal to 8. What I'm going to illustrate right now is for one such
4:44
input token but this entire visual workflow will follow for all of the input tokens. That's one key thing to
4:51
understand. For every single input token, we are going to predict three tokens into the
4:57
future. So let's say we look at I equal to0. So we look at first input token which has which is an eight dimensional
5:03
vector, right? Uh and we are predicting for depth equal
5:10
to three. So the depth is given by a variable called k and that is the same
5:15
variable which is used in deepsek paper. So k prediction depth. So we are
5:20
predicting for k= 1 we are predicting for kal2 and we are predicting for k= 3
5:25
for this uh input token. Now remember for every prediction depth so for k= 1
5:33
for k=2 as well as for k= 3 we are going to need two inputs. The first input is
5:39
something which is called as the hidden state at that prediction depth and the second input is the input embedding at
5:46
that depth. So input embedding is much more easier to explain because it just
5:51
the input embedding at that position. Right? So if I'm looking at I equal to0, I'm making my predictions for three
5:57
future positions. Right? So I'm making my predictions for I = 1 for I= 2 and
6:05
for I = 3. So the input embedding for k equal to 1 for my prediction at depth 1
6:12
the input embedding will be at position one will be which will be at i equal to 1. For k equal to 2 the input embedding
6:19
will be at position 2 which will be at i equal to 2. And for k equal to 3 the input embedding will be at position
6:25
number three which will be at i equal to 3. So that's how you get the input embedding for every prediction depth.
6:31
But there is a second uh input which you need for getting the prediction at every
6:37
depth and that's something which is called as the hidden state for k equal to 1 which is the first depth. The
6:43
hidden state is just going to be the uh output after the transformer blocks.
6:49
Right? So these input tokens are passed through these multiple transformer blocks and then we get these hidden
6:55
states. Right? So we have these hidden states for all of these input tokens and
7:01
the hidden state number zero is just the hidden state vector for that token. So
7:08
for my since I'm looking at I equal to0 my hidden state 0 will be the input
7:15
tokens when it passes through the transformer blocks what's the first row which comes out I equal to0. So that's
7:20
my hidden state vector. So for the first prediction depth K equal to 1, I have my
7:25
input embedding at this second position I equal to 1 and I have my hidden state which is the output when this first row
7:32
passes through all the transformer blocks. So these are the two inputs the h input embedding and the hidden state.
7:39
Now there are a bunch of operations which happen within every head. So when I look at head number one right now, let
7:46
me uh show this to you with a different color. When I look at head number one
7:51
right now, there are multiple operations which happen. First of all, these two vectors are merged. The input embedding
7:57
and the hidden state is merged together. I'll show you the details of this calculation later. But first, we have
8:02
this merging operation. Then we have a projection operation. Then we have a transformer layer. And after this merge
8:10
matrix goes through all of these, we have the first hidden state. Right? And
8:15
for the prediction at at depth equal to two you remember I mentioned that we need two inputs. So first input we
8:22
already have the input embedding. The second input is this hidden state which comes in the first depth calculation. So
8:28
this hidden state number one is the input for depth two. And similarly when you do the calculations for depth two
8:35
here you'll get hidden state two. That will be the input for the third token
8:40
prediction. Right? So you see the different uh token predictions or the predictions at different depth are
8:46
linked to each other in this way. The hidden state one goes as the input for
8:51
the prediction of the hidden state two. The hidden state two serves as an input for the next step prediction. And once
8:57
you get these different hidden state one, hidden state two and hidden state number three, you pass it through the
9:02
logics matrix. The logix's matrix is the same one as here which projects every hidden state into the vocabulary
9:09
dimension. And that is how we predict the next token. I'm going to explain all these steps in a lot of detail to you.
9:16
So do not worry here. I I just want to give you an overall overflow of what
9:21
exactly is multi-token prediction to. So to summarize what I just mentioned, we
9:26
have to focus at one input token at a time. So I'm looking at I equal to0 and the depth of my prediction is equal to
9:33
3. So I have to make predictions at I= 1, I= 2 and I= 3. So I need three heads.
9:39
I have head number one, head number two and head number three. In each head, I need two inputs. I need the input
9:46
embedding at that position and I need the hidden state at that position. Getting the input embedding is easy. You
9:52
just look at the future depth and you get the input embedding. You get the vector uh of the input embedding at that
9:58
position depth. So that much is easy. But getting the hidden state is not that easy to get
10:04
the hidden state at different positions. to get the hidden state at K equal to 1, you just pass the input input embedding
10:12
these input embeddings into the transformer blocks and then you get the final hidden states. Right? So since I'm
10:19
looking at I equal to0, the first row over here is going to be the hidden state zero at the head number one. Then
10:26
you might ask how do we get the hidden states for K=2 and K= 3. The hidden
10:31
states for K=2 and K= 3 come from the previous depths. So the hidden state for k=2 comes from the hidden state 1 and
10:39
hidden state for k= 3 comes from hidden state 2. And how do we get hidden states
10:44
1 2 and three? In every head we do a series of operations. We do a merging operation. We do a projection operation.
10:51
We pass it through a transformer layer. And that's how we get these different hidden states. These hidden states will
10:56
of course have dimension of 1x 8, 1x 8 and 1x 8. But when they pass through
11:01
this logits matrix, they will have a dimension of one by let's say 50,000 if the vocabulary size is
11:08
50,000. And then we'll predict the next token. We'll predict the next three tokens and then we'll compare it with
11:15
the actual three tokens at these depths and then we'll get the loss function between the predicted three tokens and
11:21
the actual three tokens. That's how the multi-token prediction works in practice. So for depth k equal to three,
11:28
three tokens are predicted for every input token. So if you look at how this is mentioned in deepseek paper, it's
11:34
quite confusing because there are two variables I and K. So I is the input token which you're looking at. Currently
11:40
we are looking at I equal to0, right? And K is the prediction depth at which
11:46
you look at for every input token. So that's how you get these three tokens. And here I have just mentioned a
11:52
small note that here we have input sequence length equal to 8, prediction depth equal to 3. This means we have
11:58
input tokens index from 0 to 7 and for each token position you
12:04
predict the next three tokens I + 1, i + 2 and i + 3. Now make sure that due to
12:11
the sequence boundary predictions are only made for positions I = 0 1 2 3 4
12:17
because at position I = 5 we can only predict 6 and 7 since there is no token
12:22
at position 8. So we have I = 0 to 7. Right? The predictions can only be made
12:27
till I = 0 1 2 3 4 0 0 1 2 3 4 because
12:34
for every input we need three future depths. So if this row is the input, we
12:39
don't have the third depth at all. So the prediction cannot be made at third depth. So the predictions are only made
12:46
for I equal to 0 1 2 3 and 4 due to the sequence boundary. That's an important thing to
12:51
note. I'm soon going to mention to you the operations which happen in different heads and I'm going to show you the
12:58
details of this merging operation, the projection operation, the transformation, the transformer layer
13:03
etc. But before that, let's take a quick look at the at an information mentioned
13:08
in the deepseek paper. What Deep Seek specifically mentioned is that they sequentially predict additional tokens
13:15
and keep the complete causal chain at each prediction depth. What does this mean? What does this mean? Keep the
13:20
complete causal chain at each prediction depth. What they mean is that one one
13:26
prediction influences the other predictions. Right? This hidden state these hidden states are passed on from
13:32
one pro projection from one depth to the other depth. So hidden state one is passed to the second pro second
13:38
prediction hidden state two is passed to the third uh third prediction etc. So
13:43
there is causality which is maintained. Every token every next token is not calculated independently. The predicted
13:50
token one predicted token two and predicted three are not calculated independently from each other because
13:55
there is this link between the hidden states. Right? The hidden state one is passed for calculation of the predicted
14:01
token two. The hidden state two influences the calculation of the predicted token three. So there is a
14:07
causal chain between the next tokens which are predicted. And this is very different from the first paper which
14:13
introduced multi token prediction. And if you if you remember the first paper which I showed you in the previous
14:20
class, this was the paper which came from meta in April 2024 which first
14:25
implemented multi-token prediction and they predicted the next tokens independently from each other. So as has
14:32
been mentioned here the first paper uh which I believe is the same paper over here they predicted uh additional tokens
14:40
using independent output heads. So deepse differed in that aspect. Deepseek
14:45
made sure that there is quasality between the next tokens which are predicted. That's one main innovation
14:52
which Deepseek added to the multi-token prediction. And this is one common thread in all of the deepseek
14:57
innovations. They took uh inspiration from what people have done and they just
15:02
made it better. They innovated on top of that. So for example, this hidden state passing from one projection depth to the
15:09
other projection depth that was not present in the original multi-token prediction paper. In the original paper,
15:15
the next tokens were calculated independently from each other, but deepc completely changed that. That's why they
15:22
have mentioned that a complete causal chain is maintained and they claim that this improves their results, which I
15:29
also think it might be true because the information from the past prediction goes into the future
15:35
predictions. So this is the whole idea of how deepseek implemented the multi-token prediction. But now I'm
15:41
going to show you what is this merge matrix. What is this uh uh what is this
15:46
merge matrix? What is this uh projection matrix? And what is the transformation
15:52
transformer layer? How do they exactly work in practice? So now let me show you the exact operations which happen in
15:58
every head. All right. So let's now dive into every head with the magnifying
16:04
glass and let's try to understand the operations which are happening inside every head. Okay. So I'm going to show
16:11
you the operations which happen over here. So let me first rub some of these
16:16
things over here. I'm going to show you the operations which happen at the first
16:21
prediction depth over here in head number one and then you will get a fair idea of what the operations are which
16:28
happen in different heads as well. Right? So when we are looking at head number one uh there are two inputs the
16:34
input embedding at the first position and the hidden state zero. Right? So the input embedding at the first position
16:40
will be of the dimension 1 by8 and the hidden state for this I equal to0 it
16:46
just means passing the input tokens through the chunk of transformers and then taking the first row of the output.
16:52
So this will also be 1x 8. So as you'll see here the head number one receives two inputs the hidden state
16:59
which is 1x 8 and the input token embedding at the next position which is at i equal to 1. So this is also 1x 8.
17:07
So the first operation which we do is we merge the hidden state and the input token embedding. So we merge the 1x8
17:14
vector with the 1x8 vector and that leads to the 1x6 vector. Okay. Uh however before we do
17:23
the merging operation as you'll see over here before we do the merging operation we take an RMS norm which means uh we do
17:30
the root mean square normalization of these vectors. we do the root mean square normalization of this vector and
17:36
this vector. So that is an important step. We do the RMS norm of this vector and we do the RMS
17:43
norm um we do the RMS norm of both of these
17:48
vectors before concatenating them together. So we do the RMS norm here and then we join them together. That's
17:54
what's shown over here. Uh the RMS norm of these vectors and then we join them together. That's
18:01
the first step. Okay, after you merge them together, after taking the root mean square normalization, first of all,
18:08
if you are not aware of the root mean square normalization, it just dividing by the
18:17
uh mean of the squares and then taking a square root of that. So, RMS norm is a
18:22
very common averaging technique. So, if you are not aware of that, uh you can just look up RMS normalization averaging
18:28
technique. It's a pretty simple normalization formula which is a bit different than layer normalization. In
18:34
layer normalization, you subtract the mean and divide by the square root of the variance. Right? Here it's a bit
18:41
different. Um here you just directly divide by the square root of the average of the
18:48
squares values. Um so now after you get the merged embedding which is 1x 16 you
18:54
multiply it with the linear projection layer which is a matrix of dimension 16a 8. So then 1x 16 * 16a 8 that is a 1x8
19:04
vector. So in the deepsek paper this multiplication with the projection is
19:10
this matrix is denoted by mk. So mk is the projection matrix which I just
19:15
showed you and its dimensions are da 2 * d. uh so here I'm just looking at one
19:20
token that's why it's 1 * 16 but if you look at all the eight tokens it will be 8a
19:26
16 so for one token it's 1x 16 uh sorry
19:32
actually linear projection is d comma 2D itself right so 2 * 8a 8 so if you multiply this with 16a 8 this will give
19:39
you 1x 8 so that makes sense now after so there are two operations here right
19:44
there is a merging operation and there is a projection operation which we saw the merging operation ation and the projection operation. After this, we
19:51
have a vector which is 1x8. So, it's the same dimension as the uh input token embedding and the hidden state. And this
19:58
1x8 vector then is an input to the transformer block. So, we have one more
20:03
transformer block here similar to the transformer block which is present in the original LM architecture and
20:09
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

