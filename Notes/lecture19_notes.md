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
* __Step-4__: How much weitage needs to be given to each expert? This is decided by __routing mechanism__.


***

* 25:00

* 
* __Step-5__:
* __Step-6__:
* __Step-7__:


***

* 30

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








