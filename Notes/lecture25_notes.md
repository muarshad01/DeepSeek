
0:05
Hello everyone and welcome to this lecture in the build deepseek from scratch series. Today we are going to
0:12
code the multi-token prediction mechanism from scratch in Google collab.
0:19
So we have finished two lectures on multi-token prediction so far. In the
0:24
first lecture on multi-token prediction, we looked at what exactly is multi-token prediction, how it differs from single
0:31
token prediction and why is multi-token prediction useful. We looked at four
0:36
major reasons for why multi-token prediction is useful. First is that it leads to densification of training
0:42
signals. It just makes pre-training a lot more efficient and a lot more
0:49
smarter. Second, it leads to improved data efficiency. Third, it's it leads to
0:56
better planning for the language model. And fourth is that it leads to higher inference speed. We also saw that DeepS
1:03
used the multi-token prediction gains only during pre-training. And during inference, Deepseek just used single
1:10
token prediction. And in the second lecture on multi-token prediction, we saw the exact mechanism
1:16
through which DeepS implemented uh multi-token prediction. So we saw that
1:22
we looked at this sequence of input tokens where there were eight input tokens each of a dimension of eight. And
1:29
then we saw that for every input token there were multiple tokens which
1:35
were predicted. So there is a terminology called as the depth of token prediction. So if depth equal to three
1:42
it means three tokens are predicted for every input token. So for example if we
1:47
look at this input token in the first row we have head one which predicts the
1:52
first token. We have head two which predicts the second token and we have head number three which predicts the
1:58
third token. And within each head there are multiple different computations which
2:04
happen. So if you zoom into one head, you have a hidden state coming into that
2:10
head and the input token embedding at that position which are merged which are then passed through a linear projection
2:17
matrix which is then passed through the transformer block and then finally
2:22
that's passed through the logits matrix which gives us the next token at that head. So we have the merge matrix,
2:29
projection matrix, transformer layer. Then we have the output of the transformer which is the hidden state
2:35
that's then passed through my unmbedding matrix or the logits matrix and I get my prediction token for the for this
2:42
head. Now I mentioned two inputs coming into each head right first is the input embedding and second is the hidden
2:48
state. The input embedding is for that prediction for that depth at which we
2:54
are predicting. So for every given input we are looking into the future right. So
2:59
for the first head we look at the next position. For the second head we look at
3:04
the position after that. For the third head we look at the position after that and we take the input embedding vector
3:10
corresponding to each of those positions. That is the first input to every head. The second input is the
3:15
hidden state to each head. And for head number one, this hidden state is just the output of the first input token.
3:23
When it passes through the different transformer blocks in the LLM architecture, we get the output
3:30
corresponding to that particular token. And that's the hidden state for head number one. But for the subsequent
3:37
heads, the hidden state is given from the previous calculation. So for example
3:42
for head number two the hidden state one which is computed in head number one that's passed as an input to head number
3:49
two. For head number three the hidden state which is computed in the previous computation is passed as the second
3:55
input along with its input embedding. So that's how information of the previous hidden state is carried forward. This is
4:03
different than how multi-token prediction is uh was implemented traditionally where every token
4:09
calculation was independent but here it's not independent right there is causality which means that we are taking
4:16
information of the past and transferring it into the future that's one main innovation made by DC. So we saw this
4:24
entire mechanism in the previous lecture and we saw the mathematical implementation. So finally after we have
4:30
the predicted tokens for every input token we take the loss function between the predicted token and the actual token
4:37
we add all of these loss functions and that gives us our total loss. If you want to revise this entire
4:44
mechanism which I just revised you can take a look at the previous lecture because now we are going to code this
4:50
entire mechanism which I described right now in Google Collab. So let us head to Google Collab
4:57
right now. So when we start to code this multi-token mechanism or multi-token
5:02
prediction mechanism from scratch uh the code is not very long actually and we just have four steps. Step number one,
5:10
step number two, step number three and step number four. Okay. In step number
5:16
zero actually we are just going to load the packages. In step number one we are going to define the RMS norm class. In
5:23
step number two, which is the longest and the most important step, we are going to just define a simple
5:28
multi-token prediction class. In step number three, we are going to generate
5:33



***


next tokens. And in step number four, I'm just going to demonstrate the loss function calculation between the target
5:39
token and the predicted token. So let's get started. First, we have to load the
5:44
packages. And in this simple demonstration, the only package which we are going to need is PyTorch. So you
5:50
you'll just need to import PyTorch and you'll need to load it. So this step will not take too much time. Let me also
5:57
run this along with you. So let me run this live so that I can demonstrate to you that this entire code actually does
6:03
not take too much time to run. Even if you are on a normal CPU um or if you
6:09
have T4 GPU that much is fine uh for you to run this code. The second or the step
6:15
number one rather is to define the RMS norm class. Why do we need RMS norm? If
6:20
you remember for every head before we concatenate the hidden state and the input token embedding, we have to take
6:27
the RMS norm of that vector. And this is also mentioned in the deep C paper.
6:33
Before we concatenate the hidden state and the input embedding for a given head, we have to take the RMS
6:38
normalization of that vector. Now, how is RMS normalization calculated? It's
6:43
pretty simple. Let's say if you have a vector uh let's say if you have a vector
6:48
with a bunch of entries right let's say if you have a vector with 6 7 8 values
6:54
what you do is that you first take the square of each of them. So you um you
7:01
take each of these entries and you square them. You take each of these entries and you square them. The next
7:07
step what you do is that you you add add all of these squared values and then you
7:12
take the uh mean which means if there are eight entries you sum all of these
7:18
values and you take the mean and then what you are going to do is that you are going to divide every
7:25
entry with the square root of this. So the root mean square is you calculate the root of this mean of the summation
7:32
of the squares and then you are going to divide every entry of this vector by this quantity. Okay, that is the root
7:39
mean square calculation and if you are unsure about this you can just check RMS RMS calculation. So this is root
7:47
mean square. uh true so to get the RMS value you just sum the squares you divide by n you take the square root and
7:54
then you normalize every entry in the vector by this quantity that's exactly what's done in this RMS norm class we
8:02
first sum the entries we take their mean we take the square root and then we
8:08
divide every entry in my vector with this RMS value there is a small epsilon value which I've added here so that we
8:15
prevent the division by zero okay so that's the RMS normalization class. The
8:20
step number two is where we define the multi-token prediction task, multi-token prediction class. And this is probably
8:27
the most important part of this code. So here you'll see that there is the init module where we have to initialize a
8:34
number of different matrices and then we have the forward method. Okay. Uh so the best way to
8:42
start understanding this code is right over here where we have these loops. we have the first loop and the second loop.
8:48
So let me exactly explain to you what is going on here. What we are basically doing is that for every input token, so
8:57
let me take you back here. Yeah, for every input token, let's say for i=0, we
9:04
have a variable which is called as k and that's my prediction depth. Right? So for i
9:10
=0 um for i =0 I'm predicting
9:17
at i + k. So I'm predicting at depth k depth
9:22
equal to 1, depth equal to 2 and depth equal to 3. Similarly, if I is equal to
9:28
1, I I'm going to predict at depth 2, 3 and four. uh if I'm predicting three
9:34
future tokens if I equal to two I'm predicting at depth three four and five
9:39
okay so the key thing to note here is that for every token position I we are
9:45
predicting the next three tokens I + 1 I + 2 and I + 3 so in the code you will see K right so this K is this value the
9:53
depth and I is the position of the token right and how many such I you need to
9:59
consider I'm looking at three future tokens Right? So I cannot have this I because the first future position is
10:06
this. The second future position is this. But the third future position is not in the input sequence at all. Right?
10:12
So in this entire input sequence I can only consider I=0, I= 1, I= 2, I= 3 and
10:19
I 4 as my input because when I= 4 is my input, my next three tokens which are
10:25
predicted are I= 5, 6 and 7. And I reach the end of my input sequence.
10:31
So the key thing to note here is that we have two indexes which are changing. The input index index index is changing from
10:38
i= to 0 1 2 3 4 and 5. So the input index I goes from 0 1 2 3 4 and 5. And
10:46
then for each of these the depth k will go from if k is indexed at zero then it
10:52
goes from 0 1 and 2. So since Python has zero indexing, K will go from 0 1 and 2
10:58
which means depth of 1 2 and three for this I. Then for this I we have again K
11:05
going from the the next three which are which is the depth of 2 3 and four etc.
11:11
So there are two loops. There is an outer loop of my index and there is an inner loop of the uh depth at which we
11:19
are predicting for that index. And that's why you'll see that in the code there are two
11:24
loops. There is first loop which is I uh and then there is the second loop for
11:30
K. Okay. And I goes from range of 0 comma max I + 1. Now what is this max
11:37
I? Max I is T minus number of heads minus one. What does this mean? So
11:44
essentially T is the input sequence length. Number of heads is how many how much depth you are predicting into the
11:50
future. Why we are doing this is because in this case in this case t is equal to 8 right because I have eight input eight
11:57
input tokens in this sequence. Then the number of heads number of heads
12:03
is equal to three correct because I have head one head two and head three I'm
12:08
predicting three future tokens. So t minus number of heads t minus number of heads is equal
12:16
to five and I can only vary i from 0 1 2 3 and
12:23
four. As you have seen I can only take five of these rows. I cannot take the sixth row because the ne the third token
12:30
will be out of reach. That's why you have I going from max I + 1 which is t
12:36
minus uh number of heads and Python has the zero indexing system right so this
12:41
will go from 0 1 2 3 and four in our particular case okay so that's where
12:46
this outer loop comes from and the inner loop k is in the range of number of
12:52
heads which means 0 1 and two because I'm predicting three future depths okay now what I'm going to do is that
13:00
uh for every input. So let's say if I =0. So if I =0, let's say I = 1. First
13:09
let me show you for I equal to0. You'll see in the code that the future position is I + K + 1. The future position is I +
13:18
K + 1. And remember Python has zero indexing system, right? So K goes from 0
13:23
1 and 2. So if I =0 this will be so k=0
13:29
k= 1 and k = 2. So for i=0 and k=0 i + k
13:35
+ 1 will be 1 then here will be two and here this will be three. This is correct
13:40
right because we are looking at position number one position number two and position number three for this i. Now if
13:47
I is equal to 1, I = 1, then this will be I + K + 1
13:55
will be position number 2, 3 and 4. That's why this I + K + 1 makes sense
14:01
over here for every input um token position index. We are looking at three
14:07
positions into the future and Python has the zero indexing system. So that's why we have to do K + one over here. If you
14:14
look at the deep se I + K they don't have I + K + 1 and we have to do I + K +
14:21
1 in the code because Python has zero indexing system. So that's why the future position for every input index I
14:29
is I + K + 1 and we are looping this. So if you focus on the first if I equal to0
14:35
so let's now restrict our attention to I equal to0 okay then K will go from 0 1
14:41
and 2. So the future positions are position number one, position two and position three. So for I equal to0
14:50
uh let me rub this. So for I =0 the future positions are
14:56
position number one, position two and position number three. That's what we have written over here. For I
15:02
=0 uh the we we are looking at position number one, position number one,
15:08
position two and position number three. Okay. Now what we have to do is that we have to the first step in every head is
15:16
we have to take the input embedding at that future position and we have to take the hidden state. Now initially before
15:23
we enter the head we have to first define my edge previous which is my initial hidden state and this initial
15:29
hidden state will be the uh output of my transformer block. So here I'm just
15:36
defining H0 sequence if you see yeah H0 sequence equal to embeds right. So here
15:43
I'm just defining the initial H0 sequence as from the input embedding.
15:50
But actually in a deepse architecture what they did was the initial hidden state was when you take
15:57
the input embedding for i equal to0 when you pass it through the transformer block and when you first get this output
16:04
that's the initial hidden state which we which you have to initialize over here that's my h previous okay and then when
16:11
you go into each head now for k in range number of heads which means I am at head number zero or head number one 2 and
16:18
three in each head I have to take two inputs right? I have to take my token embedding and I have to take the
16:25
previous hidden state. As you see over here in each in each head you have two
16:31
inputs. The input number one is the input embedding at that position and the hidden state. So what you are going to
16:39
do is that first you get the token embedding which is the input embedding at that position. So that's why we have
16:44
future position I + K + 1. So you get the input embedding at that position and
16:50
then what you do is that you take the RMS norm. You take the RMS norm of the hidden state and you take the RMS norm
16:56
of the token embedding at that position. And this I have also shown over here. You take the RMS norm of the hidden
17:03
state. You take the RMS norm of the input embedding at that position and
17:08
then you merge them together. That's what's shown over here. you take. So this H norm is the RMS norm of the
17:14
hidden state to that head. E norm is the RMS norm of the token embedding at that
17:20
future position. And then we merge these two together. So that gives us the merged vector. And the merged vector
17:27
will now be 1x 16. So if this is 1x 8 and this is 1x 8, the merge vector will be 1x 16. So you have this merged
17:34
vector. Then what you do is that once you get the merged vector, you have to project it. Uh so my merge vector is 1x
17:41
16 right I need to project it back to 1x8 vector so I need to multiply it with a linear projection layer whose
17:48
dimensions are 16 comma 8 so if you see here selfp projections that would have
17:54
been defined in my init so this is the selfprojection and you'll see it's 2D
17:59
comma D so the selfp projections is a linear neural network layer which is two
18:06
times the dimension model uh comma the D model, right? Uh so this is the
18:14
dimension of the projection layer which we have and the when you uh pass the
18:20
merge matrix through the projection layer, the output is of the same as the uh dimension. So it we get a 1x8 vector
18:27
over here. So this 1x8 vector which we obtain after the projection layer is
18:32
then passed through the transformer block. So to simulate the transformer block
18:37
what we have done here is that uh we have we are just uh using this
18:42
transformer encoder layer. So if you scroll up you'll see that transformers
18:48
is just transformer encoder layer and this is a functionality which is given by pytorch which automatically input
18:55
which automatically does all the calculations as in the forward pass of a transformer. So the transformer encoder
19:02
layer involves multi head self attention layer normalization feed forward neural network and another layer
19:09
normalization. So by default you can use this uh package from the from PyTorch.
19:15
So this is n.t transformer encoder layer. So this projection is eventually
19:22
passed through the transformer block and then what I get is I get the hidden
19:28
state for the current head. This I'm calling H C U R which is H current and
19:33
uh this is H previous. So if you take a look at the schematic now um for the
19:39
first head for this first head what I'm doing is that this hidden state I'm calling as
19:45
edge previous in the code and this hidden state which is the output of the transformer block is called as the edge
19:51
current in my code and this current hidden state is then passed through the
19:57
unmbedding matrix u which gives me my logics matrix
20:03
um and This gives me my logits which means that it will help me in prediction of the
20:10
next token. So for the first head for the first head the dimensions of this
20:16
the this output is actually 1 comma the vocabulary size. So if you see over here this b is
20:24
the batch size. So for all practical purposes you can assume the batch size to be equal to one. Uh so we get the
20:30
output which is equal to 1 comma the vocabulary size uh for the first
20:36
prediction depth. Then what you are going to do as you are going to loop for different prediction depths you are
20:42
going to merge these logics together. So finally the logits which is accumulated
20:49
for one input index. So if you look at this I =0. If you look at I equal to0 we
20:56
have three logits vectors which are predicted right 1 comma vocabulary size then 1 comma vocabulary
21:03
size and 1 comma vocabulary size. So if you aggregate these three
21:10
together for one index uh for i equal to0 you get a logit vector which is 3
21:16
comma vocabulary size. And similarly you are going to make these predictions for i= 1, i= 2, i
21:25
= 3 and i = 4. Correct? Because we are going to predict the next token until
21:30
this point until this i i equal to0 1 2 3 and four. So for each of these we'll have
21:38
the uh final logits as 3 comma the
21:44
vocabulary size which are the next three tokens which are predicted. So we get
21:49
these logit vector for the first input token we have the three comma vocabulary
21:56
size for the second input token we have the same size etc. So in the code this logit k which I have shown over here is
22:05
from we are first doing one input token we are then predicting the logits vector
22:11
for all the prediction depths and we are stacking it together. So for one input token there are three prediction depths.
22:17
So when you stack it together for one input token you will get three comma vocabulary size and then this output now
22:24
this output is the 3 comma vocabulary size done for
22:29
five such types. So this output now when you stack all of this together this will
22:35
be 5a 3 comma vocabulary size and here I'm assuming the batch
22:42
size equal to 1. So if you also take the batch size this will be of the size B comma 5a 3 comma the vocabulary
22:52
size this is the output which comes out and the way to read this output is that
22:57
for every index uh for every input token index we are
23:03
predicting three tokens into the future so that's three comma vocabulary size and how many such input indexes are
23:08
there there are five such input indexes and how many batches are there there are batches So the output dimension is BA 5A 3A
23:17
vocabulary size. What is this five? This five is equal to the input sequence T minus the prediction depth T. So it's 8
23:25
minus 3 which is equal to 5. This three is the number of heads which I have or my prediction depth. B is the batch size
23:31
and vocabulary size is whatever vocabulary size which we have in our language model. So here you see the
23:37
output sequence is B comma T minus B comma D comma V. So B is the batch size.
23:45
T minus D as I mentioned is the sequence length 8 minus the number of heads which is three. So this is five. D is equal to
23:52
three which is the number of heads or the depth and V is equal to the vocabulary size. Okay. So this entire
23:59
code right now which is the simple MTP class is the main module
24:05
which we have implemented. In this what we did is that first I mentioned to you
24:11
that there are two loops. There is a loop for sliding over my token index and
24:16
there is a loop for sliding over the prediction depth. So for each token index we are actually predicting for
24:22
three tokens into the future. If K is equal to three, uh then we are first aggregating the RMS norm of the previous
24:29
hidden state and the token embedding. We are merging it together. We are passing it through the projections matrix so
24:35
that we we are brought back to the model dimension. Then we pass it through the transformer layer to get the current
24:41
hidden state. This current hidden state is passed through the unmbeding matrix or the logics matrix and then we get one
24:49
or the vocabulary size vector for that prediction depth. And we have three such prediction depths for every input
24:55
position. So this logit scale for every input position will have 3 comma vocabulary size. And we have five such
25:02
input positions. So it will be 5a 3 comma vocabulary size. And if we also consider batch size then it's b comma 5a
25:10
3a vocabulary size. That's the output which we get. So the best way to
25:15
visualize this output is that for every input sequence I have
25:21
three prediction tokens which are the output. Now for i=0 if you consider
25:27
uh for i=0 I have my prediction at i = 1 2 and 3. If you consider for i= 1 I have
25:35
my predictions for i= 2 3 and 4 etc. Right? So we have actually this this a
25:42
fourdimensional tensor consists of three prediction tokens as output for five
25:48
input sequences. That's the main thing or that's the main way to visualize it. This fourdimensional tensor consists of
25:56
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

