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
* Expert Selector Matrix = Input Matrix  X Routing Matrix


***

* 25:00

* __Step-5__: Expert selector weight matrix.


***

* 30:00

* __Step-6__:
* __Step-7__:


***

* 35:00

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






***

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











