* FFNN - (Expansion, Contraction) network
* MoE - Multiple NNs

***

5:00

* Reduce pre-training time
* Accelerate inference
* Each Transformer block has different experts

* Sparsity
* Routing


* __Step-1__: We start with an input matrix of size (4,8) as below.
* __Step-2__: When we pass the input matrix throught these three experts, we get three output expert matrices.
* The challenge is how to merge these three expert output matrices such that we get a resulting output (4,8) matrix
* __Step-3__: To do the merging, we use a mathod called "load balancing".
* __Sparsity__: How many experts will be active for each token?
* Every token will be sent. to a selected nymber of experts. This is called "load balancing".
* __Step-4__:
* __Step-5__:
* __Step-6__:
* __Step-7__:

block. row for a given token there might be different experts activated in different transformer blocks. So let's
6:05
say in the first transformer block this is the uh expert which learns about um
6:12
worbs in the second transformer block this is the expert which learns about verbs etc. So this is the visual journey of a
6:19
token which we saw in the previous lecture as it passes through multiple transformer blocks.
6:24
Today my main intention is to show you step by step how the mixture of experts is actually implemented which means that
6:31
given any token let's say a token comes in with a dimension 768 right after it
6:38
passes through all of my experts it should still retain its dimension of 768. So what happens in this process how
6:45
do we utilize all of my experts to ultimately give my output which has the same dimension as that of my input.
6:53
While we do this, we'll we'll see a number of ideas. We'll see the idea of sparity, which means that not all of my
6:59
experts will be activated at the same time. So that's one main concept which we are going to see today. And the
7:05
second concept which we are going to see today is routing. Routing means that it's a
7:13
routing mechanism which decides based on a given token which expert the token
7:18
needs to be routed to. So these are the two main concepts which we are going to see today and we are going to look at
7:24
this concept in a sequence of steps. So we have step number one, step number two, step number three, step number
7:30
four, um step number five, step number six and step number seven. So we are
7:37
going to look at these seven steps today sequentially and I have this lecture planned out in a completely visual
7:42
manner so that you will understand exactly how mixture of experts is implemented. So let's get started with
7:48
this sequence of steps. All right. So let's get started with step number one initially and this is the most simple
7:55
step. In this step we essentially start with my input embedding matrix. So I have four tokens the next day and is and
8:02
each of them have a dimension of 768. So each of these token has gone
8:08
through an entire journey before reaching this speed for our neural network stage. Right? Each neural each
8:13
token is tokenized converted to token embeddings. we add positional embeddings. Then it goes to the
8:19
transformer. We have a layer of normalization. We have multi attention, dropout, another layer of normalization
8:25
etc. So after going through all of these sequences, I have those input embeddings
8:31
and now those are the inputs to my feed forward neural network here. Now instead of one feed forward neural network, we
8:37
are going to see how the mixture of experts block is implemented with multiple neural networks. All right. So
8:43
this is my input matrix with four tokens and each of the token has an embedding dimension equal to
8:49
8. What I'm going to do next is that we are going to pass this input matrix to a mixture of experts model with three
8:55
experts. So in this step which we saw instead of passing these tokens to a
9:00
single neural network, we are going to pass it through a mixture of experts model which has three experts or which
9:06
has three different neural networks. So let's see how this is implemented. Our main idea is that let's say you have
9:13
this input embedding matrix and I have my expert number one, expert number two and expert number three. Each expert is
9:20
a neural network which is an expansion contraction neural network which retains the input dimension. So when the input
9:27
embedding matrix is passed through the expert E1 U the dimension of each token
9:33
initially is 768 and it's retained or let's say the dimension of each token is equal to 8 and that's retained to be
9:40
equal to 8. Right? So whenever the input embedding matrix is passed through the
9:45
feed forward neural network E1 which is expert one we get the expert output one
9:50
which is a 4x8 matrix. Why is it 4x8? This will be relevant to you later. So
9:56
please try to understand the intuition behind what every row and every column
10:01
here means. What does every row in the expert output matrix mean? Every row essentially corresponds to tokens,
10:08
right? The first row corresponds to first token which is the the second row corresponds to the
10:14
second token which is next. The third row corresponds to the third token which is day and the fourth row corresponds to
10:20




***

is and 8 is just the dimension of the input which is retained. Similarly, when the input embedding matrix passes
10:26
through my second neural network E2, I get my expert output matrix number two
10:31
where again the first token is the, the second is next, the third is day and the
10:38
fourth is is right. And similarly when my input embedding matrix passes through E3 I get my expert output three where
10:46
again the first token is the second token is next third token is day and the
10:51
fourth token is is. So what I have written here when we pass the input matrix through these three experts we
10:58
get three expert output matrices as shown in the above figure. We get expert output matrix 1 2 and three. The
11:05
challenge is that we have three output matrices each of the dimension 4x8 but
11:10
we only need one matrix which is the output matrix which has the dimensions equal to 4x8 right I don't want three
11:16
4x8 matrices as the output of my mixture of experts model I somehow want to
11:21
combine the output of all of these expert matrices so that I get a resultant matrix which is
11:27
4x8 and once you formulate this question this way that's when the whole journey
11:33
of mixture of experts really starts S we have three experts. Now each of the expert is contributing to something.
11:39
Each of the expert has produced a 4x8 output. Now we need to decide how to combine these matrices together. Combine
11:46
these three matrices, three expert output matrices together so that I get a single output matrix which has the
11:52
dimension equal to 4x8. So let's see how this is done. The
11:58
first thing which we are going to implement is called as parity.
12:03
To do this merging of these three matrices, the first thing which we are going to implement is sparity. That
12:09
means that we are going to decide how many
12:15
experts how many experts will be active for each
12:25
token. We are going to decide how many experts are going to be active for each token. Remember I mentioned to you at
12:31
the start of the lecture that in mixture of experts one token will not be passed
12:37
to all my experts. We are only going to pass every token to only a certain set of experts. So in this case let me
12:44
decide that out of these three experts I'm only going to activate two experts
12:50
for every token. I don't know what those two experts are and I don't know which of these three I should activate. But I
12:57
only know that every token let's say I look at the tokens the uh it will not be
13:03
passed through all of my uh experts. It will only be passed through two experts.
13:08
I will tell you how these two experts are selected but I will not pass any token through all the experts. I will
13:15
only pass it through two experts. This is also called as load balancing.
13:20
So what I have written here is that every token will only be sent to a
13:26
select number of experts. This is called as load balancing as shown in the above
13:31
figure. Let's say we decide that every token will be routed to only two experts. That's the major decision which
13:38
we have taken in step number three. And that's the decision which I'm calling as sparsity
13:43
decision. Usually the sparity can be much higher. So if you have 64 experts, you can decide that I'm only going to
13:50
make two experts active for every token. There the sparity is so high which means you have only two experts active and 62
13:57
experts inactive. But in this case I have two experts active and one expert inactive. Okay, that's called load
14:04
balancing where I implement sparity. But still that does not tell you how I'm
14:10
going to merge all these three matrices to get my final 4x8 matrix. Right? Uh be
14:15
patient. I'm going to come to that part in a moment. But this is the first concept which we have learned about
14:22
today that sparsity which is essentially selecting how many experts we want to route every token to. The second concept
14:28
which we have to now learn is called as routing. Okay. So once we have decided
14:34
how many experts will be assigned to each token, the next question which we have
14:40
to decide is how much weightage needs to be given to each expert.
14:46
Which means that okay you have decided that uh you are going to assign two
14:51
experts right for every token. So let's say the token is next and I have three
14:57
experts expert one expert two and expert three I have to decide firstly I have to decide which of these three experts
15:05
which of these three experts I'm going to route next to maybe it's E1 and E2
15:10
maybe it's E2 and E3 or maybe it's E1 or E3. I have three choices, right? That's the first thing I have to decide. The
15:17
second thing which I have to decide is let's say if I select E1 and E2, how much importance should I give to E1 and
15:23
how much importance should I give to E2? Maybe E1 is the expert which deals with
15:29
words, right? And E2 only deals with numbers. So I need to give high weightage to E1 and not that high
15:34
weightage to E2. So I need to decide two things. I need to decide which experts which which two experts that's
15:43
the first thing which I have to decide and the second thing I have to decide is how much weightage to each
15:51








***

expert how much weightage to be given to each expert and this these two questions
15:57
I have to address basically for all of my input tokens and there is only one
16:02
matrix which answers these two tokens and that matrix is called as the routing matrix. mat and this mechanism which we
16:09
are looking at right now it's called as the routing mechanism. So the routing matrix is
16:15
again a trainable matrix. If you have the input matrix which is of the size 4x 8, we'll be multiplying it with the
16:22
routing matrix and the size of the routing matrix is the the number of rows in the uh routing matrix is basically
16:29
the number of dimensions which is equal to 8 in this case and the number of columns in my routing matrix is equal to
16:35
the number of experts which I have. Okay. So since the number of experts is going to be three, the number of columns
16:42
which we have in the routing matrix is equal to three. Right. So I'm going to multiply the
16:49
input matrix with the routing matrix and then I get a matrix which is called as expert selector matrix. The number of
16:56
rows in the expert selector matrix is equal to four corresponding to the tokens. This is the first token the
17:02
next the next day is and each column here this is the most important thing
17:08
each column corresponds to one expert. The first column here corresponds to expert number one. The second column
17:14
here corresponds to expert number two. And the third column here corresponds to expert number three. We are now going to
17:20
see we are now going to see how this expert selection matrix helps us to answer these two questions. So let me
17:27
copy this thing over here. These two questions which we have formulated and let me paste it over here. We are going
17:33
to see that the expert selection matrix is going to help me answer both of these
17:38
questions. It will help me answer which two experts I have to select for every token and how much weightage I have to
17:43
give to each selected expert. Let's see how the expert selector matrix answers these questions. Okay, for example, if
17:51
the expert selection expert selector matrix looks something like this. Let me write this over here. This is my expert
17:57
selector matrix. Okay, it has four rows
18:02
because I have four tokens and it has three columns. The first job of the expert selector matrix is to decide this
18:08
thing which we have written over here. It has to decide which two experts I have to route my token to. Right? So to
18:15
do that what it does is that it it takes a look at every token right. So I'm going to look at the first
18:22
row which is my token number one which is the so the next day is I'm going to
18:29
look at my first token and I have to decide which two experts I want to route this token to. So I take a look at all
18:35
the values of my first row and I only keep the two highest values. So the two highest values correspond to expert two
18:42
and expert three. Right? So token number one is going to be routed to expert number two and expert number three. Then
18:48
I look at token number two which is next and the highest values correspond to expert number one and expert number
18:54
three. So token number two is going to be routed to E1 and E3. Then I look at
19:00
my next token which is day. The highest values are for E2 and E3. So the next
19:06
token which is day is going to be routed to expert number two and expert number three. and my final token which is is
19:12
has the highest values for E1 and E3. So my final token is going to be routed to expert number one and expert number
19:19
three. So just by looking at the values of every row, uh I can now decide which
19:24
two experts I'm routing every token to. So this question is answered which two
19:30
experts I have to route every token to. Then I have to decide how much weightage
19:35
needs to be given to every token. Right? should I give 40% weightage to to this 60% weightage to this etc. So
19:42
essentially what I want to do is that whenever I look at every row so let's say when I look at the first row I know
19:48
that expert number two and expert number three are the experts which the token the will be routed to but I want to know
19:54
whether 40% importance needs to be given to E2 60% importance needs to be given
19:59
to E3 etc. To do that what I want to do is that I want to make sure that every
20:04
row essentially sums up to one. Every row sums up to
20:11
one. What this will help me is that it will help me make interpretable statements like if the second row sums
20:18
up to one, I can say that when next is the input give 80% weightage to E1 and 10
20:25
20% weightage to E3. Let's see. Um, right now I cannot do that because these
20:31
are just random numbers, right? I cannot just assign percentage or probability values to each of these rows. So to do
20:39
that, we have to so to put that in mathematical context, what we have to do is that we have to normalize the rows of
20:45
the resulting matrix so that each row sums up to one. And uh I'm going to or
20:51
you can pause the video for a moment here and think about which is the technique which you usually use to make
20:58
this normalization. If you are given a certain set of elements, right? If you are given a certain set of elements like
21:04
5 1 2 3 4 5. What is that operation which you will perform on these set of
21:10
elements so that all of these values sum up to one and each value will be between 0 and one.
21:17
You can pause this video to think about this. But the operation which you do on these elements is the softmax operation.
21:23
Which means that each element will be now replaced with let's say the first element is x1. The first element will be
21:29
replaced with e to x1 divided by summation of e to x1 + e to x2 plus dot
21:36
dot dot e to x6. The first element will be replaced by this. The second element
21:42
will be replaced with e to x2 divided by this summation. Similarly, the last element will be replaced with e to x6
21:49
divided by this summation. That will ensure that all of these values will now
21:54
sum up to one and each of them will also lie between 0 to 1. This is the softmax
21:59
operation which we are going to implement over here. So what we essentially want to do is that we want to achieve two things. I want to make
22:06
sure that the experts which are not selected those values are zero and for the experts which are selected those
22:12
values sum up to one. So to do that what I'm first going to do before applying softmax whatever values are not selected
22:20
I'm going to replace them with negative infinity. So here I replace a negative infinity here I'm going to replace with
22:26
negative infinity here I'm going to replace with negative infinity and here I'm going to replace with negative
22:32
infinity and then I'm going to do soft max. Why am I replacing with negative infinity? Because when you do soft max
22:39
you use the exponential operation right and an exponent of negative infinity is going to be zero. So when I apply soft
22:46
max to these values which are now replaced with negative infinity all of these values which are negative infinity
22:52
are going to be replaced with zero. That's the power of the softmax
22:58
operation. Right? This is the first thing which softmax achieves. The second thing with softmax achieves is that
23:04
every row if you look at the experts chosen in every row they will sum up to one. So if you look at the first row
23:11
right now you can say that give 60% weightage to E2 and give 40% weightage to E3. When the first token is
23:18
considered next when the sec the the when the second token is considered which is next give 90% importance to E1
23:26
give 10% importance to E3. When the third token is considered the uh day,
23:32
give 40% importance to E2 and 60% importance to E3. And when the fourth
23:37
token is considered is give 50% importance to E1 and 50% importance to E3. So now we have answered this second
23:45
question over here. Remember we started with two questions, right? Which two experts and how much weightage to each
23:50
expert? We have answered this second question. How do we know how much weightage to be given to each expert?
23:56
Basically we just look at these softmax values and we know how much weightage to
24:01
be given to every expert. So this is how the uh expert selector matrix actually
24:07
helps us answer these two questions. The expert selector matrix helps me to understand which which experts I need to
24:15
route every token to and how much weightage do I need to give to each expert. So now we are at a stage where
24:24
uh we uh we have now we are at a stage where we have
24:30
these input tokens. We have these input tokens which we started with and I know
24:36
which expert each of these input tokens needs to be routed to and how much weightage needs to be given to every
24:42
expert where the tokens are routed to. I know that. Now can I think about answering this question? Remember I
24:49
started this hands-on demonstration by saying that I have to take my three matrices
24:54
and I have to somehow merge them together into a 4x8 matrix. Now you have two additional
25:00
pieces of information. You know every token which two experts are it's going to be routed to and how much weightage
25:07
needs to be given to those experts. Now can you think about how you can use this
25:13
information to basically merge these three output matrices into just one
25:19
matrix. Again you can pause this video for a moment here and think about how you can use this information to solve
25:25
this merging question which we started
25:32
with. Okay. So we are at step number five. Now once we have the expert selector weight matrix we can merge the
25:38
three expert output matrices which we saw in step number two. Here we have to merge these three expert output matrices
25:47
so that it leads to one matrix right uh by assigning weight
25:53
factors. So now somehow intuitively you must already start thinking that I have
25:59
these weight factors. So somehow I have to look at every token. I have to assign weight factors to my expert output
26:05
matrices uh and might and I might probably need to do some additions. Um that's exactly
26:12
what we are going to do. So now let me explain to you how the merging is done for every token which we are
26:18
considering. Okay. So we are going to start looking at the first token which is equal to the. Okay. So what do we
26:25
know when we look at the I know that I have to select just E2 and E3 right and
26:31
I have to give 60% importance to E2 40% importance to E3 so when I go to the
26:37
expert output matrices right now let's look at these expert output matrices right and I have to only look at E2 and
26:43
E3 for the so let me rub this over here and rub all the
26:50
unnecessary things okay so I am only looking at the right now and what we
26:55
have seen so far is that we have to only look at uh experts two and experts three. So I have to only look at expert
27:02
output two and expert output three. Furthermore, since I'm only looking at the I should look at only the first row
27:08
of expert output two and first row of expert output three. How do I merge these first rows to give me one final
27:17
row? For that what I'm going to do is that I'm going to use the weightage. Remember what we saw for the gives give
27:24
60% weightage to E2 and 40% weightage to E3. So what we are going to do is that
27:30
this row this row I'm going to multiply with 6 and this row I'm going to be
27:36
multiplying with 04 and I'm going to be adding these two rows together and that
27:41
will give me 1x8 matrix that will be the merged or 1x8
27:48
row 1x8 vector that will be the output vector for the and similarly I will do
27:56
for all of the tokens. Let me explain this again because I have a different schematic below which
28:01
explains this. Okay. So let's look at the first token. The to see the
28:07
activated weight activated experts of the first token. We look at the first row of the expert selector weight
28:13
matrix. We look at the first row of the expert selector weight matrix and we see that experts two and experts three are
28:20
activated with weights of 6 and 4. As a result in this output, we are only going
28:27
to look at expert output two and expert output three because only E2 and E3 are activated for this highlighted token.
28:34
Then what we are going to do is that we are going to multiply this first row over here. Why the first row? Because
28:40
the first row corresponds to the first token. We are going to multiply the first row over here with 6. We are going
28:47
to multiply the first row over here with 04 because 40% weightage to E3 and 60%
28:53
weightage to E2. And we are going to add all of this together. So the mixture of
28:59
experts layer prediction for the first token is this final vector which is my 1x8 vector right now. That's how the
29:07
mixture of experts prediction works for any given token. You look at the token.
29:12
Here are the steps to be followed. You look at the token. You look at the expert selector weight matrix and you
29:18
look at which two experts are activated and how much weightage needs to be given to the two experts. Once you know that
29:24
you then go to the expert output matrices which we started this demonstration with. You look at those
29:31
two experts which are activated which are expert output two and expert output three. You look at only the first row of
29:36
this because I'm looking at the first token currently. You multiply the expert
29:42
output two first row with 6 and expert output three first row with 0 4 and you
29:47
will add these together and then you get one vector which is 1x8 and that's the prediction vector for the first
29:54
token. Let's do an exercise now for how the second token prediction is calculated and if you want to pause this
30:01
video here you can think about it on your own. Okay. So now the
30:07
second to calculate the second token prediction what I have to first do
30:12
remember the first step is to look at the expert selector weight matrix. I look at the expert selector weight
30:17
matrix for the second token next and I see that only experts E1 and E3 are
30:23
activated. I have to give a weightage of 0.9 to E1 and.1 to E3. Okay. Then what
30:29
I'll do is that I'll go to the expert output
30:35
uh I'll go to the expert output matrices and I'll see that um or maybe I should
30:40
do this over here. Okay, I go to the expert output matrices and I see that only E1 and E3
30:48
needs to be activated. Right? So only the expert output matrix 1 and expert output matrix 3 needs to be
30:54
activated. So I'm looking at the second token next. So I look at the second row of this and I look at the second row of
31:01
this and if you remember we saw that 90% weightage needs to be given to E1. So I
31:06
multiply this with.9 10% weightage needs to be given to E3. So I multiply this
31:11
with 0.1 and and then I add these two together that will give me
31:17
the that will give me the 8 dimensional vector for
31:23
next. Similarly, I get the 8 dimensional vector for day and
31:30
is. And this is the final 4x8 matrix which is the output of this input matrix
31:36
when it passes through the mixture of experts. So, we started this hands-on demonstration by asking how to merge
31:42
these three expert output matrices into one 4,8 matrix. Right? Now, you see how
31:48
that is done. You look at individual tokens. you see which experts are activated and you multiply every expert
31:55
accordingly with the weightage factor and then you get a 1x8 vector for every
32:00
token and then you collect all these vectors. So this schematic of how all
32:05
the tokens are processed is shown over here. You take the input matrix and
32:10
along with the input matrix you also need the expert selector matrix. And remember how the expert selector matrix
32:17
is calculated. The expert selector matrix is calculated when you multiply
32:22
uh when you multiply the input matrix with the routing matrix. You get the
32:28
expert selector matrix and then you have to uh choose the top top k1s. You have
32:33
to choose the top k experts the load balancing or the sparity which we saw. You have to choose the top k experts. In
32:41
this case k is equal to two. You choose the top two experts in every row. You apply soft max. That's how you get the
32:47
expert selector weight matrix. So to get this expert selector weight matrix which is shown over here, you start with the
32:54
input matrix, multiply it with the router weight matrix, then you select the top um experts in every row and then
33:02
you apply softmax. That's how you get the expert selector weight matrix. And once you
33:07
have this for every token, you just have to select which experts are activated. And so for first token if expert two and
33:15
three are activated you look at the first rows and then you multiply it with the weightage factor according to the
33:21
expert selector weight matrix and you add them together that gives you the resultant vector for the first token.
33:27
Similarly you find the resultant vector for the second token for the third token for the fourth token and then you
33:32
collect this entire matrix together which is my 4x8 output matrix. Right now
33:37
this is how the mixture of experts model entirely works from start to end. Okay. I hope you have understood
33:45
this lecture in a lot of detail. I deliberately had a visual explanation because otherwise it's it's difficult to
33:53
uh really understand how mixture of experts works. If you directly look at matrices, it's difficult to know what
33:59
exactly is multiplied by what. But if you do these visual matrix demonstrations then it becomes much
34:04
easier to do a quick recap of what we saw today. We have learned about multiple important concepts. Right? The
34:11
first concept which we learned about today is that of sparity which means that for every token not all the experts
34:17
are activated. I have to select which among the given experts are activated for every token. Um and that is also
34:24
called as top case selection. Basically I have to select how many experts will be activated for every token. In this
34:31
case I have chosen my top K to be equal to two. So two experts will be activated for every token. Correct? That's the
34:38
first thing which we learned. This sparity is actually what helps mixture of experts be so efficient at
34:44
pre-training and inference. This is also sometimes called load balancing. The
34:49
second thing which we saw is that sparity is not enough because we need to answer these two questions right
34:55
whenever so until now we have we have we have decided that okay for every token I
35:01
have to select two experts but still we have not answered these two two questions which two experts I have to
35:07
select for every token and second how much weightage needs to be given to each expert to answer that we have to take my
35:14
input matrix and multiply it with the routing matrix okay that gives you the expert selector matrix. The expert
35:21
selector matrix eventually has answers to both these questions. We look at the values of the expert selector matrix and
35:28
we choose only the top k values. So for every token I know which which experts
35:33
are selected, which two experts are selected and then what I'll do is that I'll replace the experts not selected by
35:39
negative infinity and do soft max. So now every row will sum up to one. So then I know how much weightage to be
35:45
given to each expert also. So now I know which experts to select for every token,
35:51
which two experts to select for every token and how much weightage to be given to each expert. Once I know these two
35:57
values, then I can go back to the start and then I can look at these expert output matrices again for every token.
36:04
Now what I have to do is that I have to first know which two expert output matrices will be activated. If the token
36:11
is the first row, I look at the first row of these expert output matrices and then I multiply it with the with their
36:17
corresponding weights and then I add them together. That gives me the predicted mixture of experts output for
36:24
one token. I do this for all the tokens all four tokens and that's how I get the predicted 4x8 output matrix. So input
36:31
matrix is 4x 8 and although there are three experts now my output matrix dimension is exactly 4x8. That's the
36:39
same as if it were if you're using a normal feed forward neural network as well. But the main trick of mixture of
36:45
experts is selecting the sparsity and second to understand how the routing mechanism actually
36:51
operates. This brings us to the end of today's lecture. This this is one of the main lectures to understand how mixture
36:58
of experts really operates. In the next lectures, we'll start looking at uh some
37:03
things which are called auxiliary loss to help improve the efficiency of mixture expert mixture of experts model.
37:09
We'll also look at something called load balancing. Um and then finally we'll
37:14
start looking at the deepseek innovations which came after the
37:20
original mixture of experts um literature was out there. Deepseek
37:26
built upon the original mixture of experts research and they implemented some novel techniques. So just as a
37:33
reminder our main vision is to ultimately understand the deepsee deepseek picture of experts
37:39
paper in which they have implemented several things such as uh fine grained
37:44
expert segmentation shared expert isolation um and version 3 also implemented
37:51
something called lossfree load balancing. We are going to see all of these things but for that it's very
37:56
important for us to first develop a foundation and understand what mixture of experts actually is. And I hope
38:02
today's lecture served as the foundational building block for that. Thanks everyone and I look forward to
38:08
seeing you in the next lecture.





