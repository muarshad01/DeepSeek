#### Code Mixture of Experts (MoE) from Scratch in Python
* [makeMoE: Implement a Sparse Mixture of Experts Language Model from Scratch](https://huggingface.co/blog/AviSoori1x/makemoe-from-scratch)
* [minGPT](https://github.com/karpathy/minGPT)

* __Step 0__: Load packages and import data

***

* 5:00

* T4 GPU
* A100 GPU

* __Step 1__: Define each expert as a NN network

* __Activation Function__: ReLu

***

* 10:00

$$Input ~Matrix \times Routing ~Matrix = Expert ~Selector ~Matrix$$

* __Step 2__: Implement the Router

* __Step 3__: Implement top-k load Balancing

* __Step 4__: Use -inf and apply softmax

***

* 15:00

* __Step 5__: Create a class for Top-K Routing

***

* 20:00

* __Step 6__: Create a class for Noisy Top-K Routing

* __Step 7__: Create the Sparse Matrix of Experts (MoE) module


***

* 25:00


* __Step 8__: Putting together all the building blocks of MoE model

***

* 30:00

* __Step 9__: Code the entire Transformer block: Part 1 (Multi-head attention)

***

* __Step 10__: Code the entire Transformer block: Part 2 (Assemble all layers)


* __Step 11__: Define entire language model architecture 

***

* 35:00

* __Step 12__: Create training and testing data

* __Step 13__: Define LLM Loss

***

* 40:00

* __Step 14__: Define training loop parameters and other hyper parameters

* __Step 15__: Initialize the entire model

* __Step 16__: Run the pre-training loop

* __Step 17__: Inference

***

* 45:00


matrix multiplications very easily when matrix multiplications come in front of you but at the same time all of you
45:14
should develop an intuition about the code and you should you should be very good at coding things from scratch and
45:21
building things from the ground up. Why am I showing these uh Google collab code
45:26
files which show you how to build things from ground up? The reason I'm showing these code files is because if you want
45:32
to do research and if you want to build upon something, if you want to build the
45:37
next mixture of experts innovation, how will you do this? The only way you will be able to do this is if you are
45:43
comfortable with the theory, you know how the theory works and you know the nuts and bolts of how the entire code is
45:50
assembled. So as an exercise what you can do is that you can take this code and you can
45:55
also uh uh maybe you can add expert capacity to this. So if you remember in
46:03
the previous lectures we have learned about auxiliary loss. We have learned about load balancing and we have also
46:09
learned about capacity factor. Right? In the code which I have shown to you capacity factor is not yet added. Maybe
46:16
you can add that. Then what you can do is that we have seen deepseek innovation right the first deepseek innovation is
46:22
this auxiliary loss free load balancing where they added this bias term uh I've
46:28
completely explained to you how this bias term works and it's a very simple
46:33
addition maybe you can take the code which I've just shown you and modify it with the addition of this bias term and
46:39
then you can do some experimentations with it you can play around with different uh settings and hopefully or
46:45
maybe that leads to a new research direction for you. Uh the third thing
46:51
maybe you can do is that uh they have the shared experts right. So Deepseek had this architecture where they had
46:58
shared experts uh which was a common set of experts and then they had routed experts as well. So
47:06
I have not shown that in the code right now. Maybe you can do that. Maybe you
47:11
can do these modifications. Right? So you can use this code as a starting point to do wide range of
47:17
experimentations. You see this is the output which I have got from 20 iterations. You can already see that the
47:22
output for 200 iterations was much better than this because here that colon structure is not identified by the
47:29
model. Uh but now the sky is the limit for you. Take this code run it for a huge number of iterations. do the
47:36
modifications as I mentioned with maybe shared experts maybe with the bias term
47:41
and hopefully that sets a new research direction for you. Thanks everyone. I hope you enjoyed these five lectures on
47:48
mixture of experts. We will now move towards more innovations in the deepseek series. Lots of exciting stuff is yet to
47:55
come such as multi-token prediction etc. So we'll see about that in the next lectures. Thanks everyone and I look
48:01
forward to seeing you in the next lecture.


