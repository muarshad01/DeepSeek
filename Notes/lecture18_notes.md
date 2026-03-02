#### Mixture of Experts

***

| Paper ||
|---|---|
| [(1) 1991 paper on Mixture of Experts](https://www.cs.toronto.edu/~fritz/absps/jjnh91.pdf)||
| [(2) ST-MoE paper (2022)](https://arxiv.org/pdf/2202.08906)||
| [(3) DeepSeek MoE (Jan 2024)](https://arxiv.org/pdf/2401.06066)||
| [(3) DeepSeek V2 (June 2024)](https://arxiv.org/pdf/2405.04434)||
| [(4) DeepSeek V3 (Jan 2025)](https://arxiv.org/pdf/2412.19437)||

***

* 10:00

* In a Mixture of Experts (MoE) model, we have multiple neural networks called "experts" in the transformer block.

#### What is the need to add multiple experts?
* There are two main advantages:
1. Adding multiple experts allow models to be pre-trained with far less compute compared to a dense model (without experts).
2. Allows much faster inference compared to a dense model (without experts)


* In a dense model, every input token passes through all the parameters(i.e., all layers and neurons).
* In contrast, a Mixture of Experts (MoE) model has multipe "experts" (think of them as sub-networks or feedforward layers), but only a small subset (e.g.,2 out of 64) are activated for any given input.

### Sparsity
* Sparsity is one of the main ideas behind MoE


***

* 15:00

***

* 20:00

* [The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)


```
What is 1 + 1 ?
```

***





three might handle numbers in the second
25:14
transformer block. We do not encode this
25:17
information a priority that's learned
25:19
based on the input
25:20
sequences. So actually this illustration
25:24
explains it all. If you look at this
25:27
input sequence, right? And if you look
25:29
at this particular token, it might go to
25:31
expert number one in the first layer or
25:34
in the first transformer block, expert
25:36
number three in the second block, expert
25:37
number two in the third block, expert
25:39
number one in the fourth block, etc. And
25:41
then finally I get my next token which
25:43
is M O. Now when the next token is
25:46
passed in, let's look at this next
25:47
token. Now it might go
25:49
to expert number two in the first block,
25:52
expert number two in the second block,
25:54
expert number two in the third block,
25:55
and expert number three in the fourth
25:57
block.
25:58
And then after I come out of my feed
26:00
forward neural network, I go through the
26:02
rest of my LLM architecture, the logits
26:04
matrix, then I get my next token
26:06
prediction. And if I look at this next
26:08
token, next, it goes through all my
26:10
transformer blocks, right? Goes to third
26:13
expert, first, second expert, third
26:14
expert, third expert. And then again the
26:17
output from here goes through
26:19
uh the output from here goes through the
26:22
rest of this neural rest of this LLM
26:25
architecture. I have my logitics matrix
26:27
and I have my next token
26:29
prediction. So this leads to my next
26:32
token prediction
26:35
task for my this uh input token. So the
26:39
point of this illustration is that when
26:41
you think of mixture of
26:43
experts normally people just think of
26:45
this one neural network replaced with
26:48
multiple neural networks, right? But
26:50
this this replacement happens across all
26:52
the transformer blocks. So I have
26:54
multiple neural networks not just in one
26:56
transformer blocks but in all
26:58
transformer blocks. So that
26:59
visualization is very important for you
27:01
to keep in
27:03
mind. All right. Uh one last thing
27:06
before I end the lecture is I want to
27:07
show you how deepsek uh implemented
27:11
mixture of experts through which papers.
27:13
So they had this paper called deepseek
27:16
which was released on 11 June 2024 which
27:19
is the paper which shows the mixture of
27:21
experts architecture. So they started
27:23
with base or vanilla mixture of experts
27:25
and they then implemented a lot of new
27:28
things such as fine grained expert
27:29
segmentation shared expert isolation
27:32
then load free they also have so version
27:35
two did not have a load three load free
27:38
loss balancing uh so version two which
27:41
came in June 2024 used this mixture of
27:43
experts architecture which was released
27:46
in Jan 2024 and version 3 which came in
27:49
Jan or Feb 2025 that further modified
27:52
the mixture of experts because they use
27:54
something called
27:57
uh they use something called load
27:59
free they use something called loss free
28:02
load balancing. So they used the fine
28:05
grain segmentation which we saw earlier
28:07
this if you see the main innovations in
28:09
the mixture of experts which deepse
28:11
introduced is fine grained expert
28:13
segmentation and shared expert
28:15
isolation. So in V3 which is the latest
28:18
which was the latest model released in
28:19
2025 and which became so famous they
28:22
definitely used fine grain segmentation
28:24
and shared experts but they also use
28:26
something called lossfree load
28:28
balancing. We are going to see all of
28:30
this eventually but first it is very
28:32
important for us to understand the
28:34
mathematics behind mixture of experts.
28:36
So in the next couple of lectures what
28:38
we are going to see is that we are going
28:40
to next dive deep into the
28:43
mathematics of how mixture of experts
28:45
actually operate. So we are going to see
28:48
traditional mixture of experts first
28:50
then we are going to see the innovations
28:52
which deepse introduced such as fine
28:54
grained expert segmentation. We are
28:56
going to see what shared expert
28:57
isolation means and then we are also
28:59
going to see how what lossfree load
29:01
balancing
29:02
is. So this is the whole plan for
29:05
understanding mixture of experts. It's
29:07
one of the most modern concepts in
29:08
language model architecture and it has
29:10
become very famous recently. So it's
29:12
very important that all of you
29:14
understand the introduction and
29:15
intuition. But as always in these
29:17
lectures I don't stop at the intuition
29:19
but I'll go through the entire matrix
29:21
multiplication calculations in a lot of
29:23
detail. So if you see I'm just scrolling
29:25
down below at the set of lecture notes
29:27
which I have planned for mixture of
29:28
experts. It's a pretty expansive topic.
29:31
So I want to cover the mathematics in
29:33
detail and I want to show you everything
29:35
of how mixture of experts work from
29:37
scratch. So stay tuned for the next
29:39
lecture but please make notes along with
29:42
me. This is a very important topic. Once
29:44
you understand this you will understand
29:46
the MLA latent attention which we
29:48
covered before and mixture of experts
29:50
which are the two cornerstones on which
29:52
the deepseek architecture rests. Thanks
29:55
everyone and I look forward to seeing
29:56
you in the next lecture.








