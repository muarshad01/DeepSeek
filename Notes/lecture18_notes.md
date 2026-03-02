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



and let's say layer number six. If there
19:50
are verbs such as falling, identified,
19:52
struggling, falling, signed, uh
19:55
designed, disagree, they are routed to
19:58
layer number one. If there are visual
20:00
descriptions like color, spatial
20:02
position, uh like blue, inner, uh over,
20:06
open, dark, blue, upper, those are
20:10
directed towards specialized experts in
20:13
layer number zero. If there are proper
20:15
names like Mart, Colin, Ken, Sam, Angel
20:20
etc. those are directed towards
20:22
specialized experts in layer number one.
20:25
If there are tokens which represent
20:27
counting and numbers, right? Like after
20:30
then 7 25 4 54 those are diverted or
20:34
routed to experts in layer number one.
20:36
So from this table we can clearly see
20:38
that experts specialize in punctuation,
20:42
conjunction, articles, verbs, visual
20:44
descriptions, proper names, counting and
20:46
numbers which clearly shows that experts
20:49
actually specialize in certain tasks
20:52
which means that out of these experts
20:54
which I'm showing on the board here
20:56
right now out of these experts maybe
20:59
this experts this expert learns to deal
21:01
with punctuations like comma then
21:03
semicolon full stop etc. Maybe this
21:06
token or this expert learns to deal with
21:08
verbs. Maybe this token learns to deal
21:11
with
21:12
numbers. Uh so this expert learns to
21:15
deal with numbers. Maybe this expert
21:17
learns to deal with conjunctions etc. So
21:20
whenever token comes in, if the token is
21:22
a punctuation, it will be routed to this
21:24
expert because it's specialized.
21:26
Whenever a token is a verb, it's routed
21:28
to this expert because it's specialized.
21:31
So now think about it, right? If you
21:32
were using the earlier feed forward
21:34
neural network, whichever token comes
21:36
in, I'm activating all my 4 million
21:39
parameters in each block which might not
21:41
be needed. If I know that one particular
21:44
expert can process my punctuations, why
21:46
do I need to activate all the other
21:48
parameters? It makes sense for me to
21:50
activate parameters of this much smaller
21:52
specialized neural network. And due to
21:55
this
21:56
specialization, I only activate that
21:59
expert and I don't activate other
22:00
experts at all. that saves me a lot of
22:03
pre-training time and that also makes my
22:05
inference a lot more
22:07
faster. So actually all the modern LLM
22:11
architectures are now using mixture of
22:13
experts. If you see Llama 4 which was
22:16
recently released
22:19
uh Llama 4
22:21
uses mixture of experts.
22:24
So let's see whether M O is mentioned.
22:29
Yeah, Llama 4 is their first model built
22:32
using mixture of experts architecture
22:35
and this model is the latest coming from
22:37
Meta right now which means that mixture
22:39
of experts is almost one of the most
22:42
modern and innovative techniques which
22:43
is used by deepseek which is used by
22:45
llama also and that's predominantly
22:48
because this intelligent use of
22:51
specialized experts right which are
22:53
sparsely activated based on my input
22:55
token that saves my pre-training time
22:58
and and that also makes my inference
23:01
faster. Okay. So, we have understood the
23:04
mixture of experts intuition. What do
23:06
these experts learn? I'll be sharing the
23:08
description to this or link to this
23:10
paper in the description below. The last
23:13
thing which we are going to look at in
23:14
today's lecture is that when we see
23:17
mixture of experts, sometimes we just
23:19
look at uh one one transformer block,
23:22
right? And we see that one of these
23:25
experts or sparse combination of these
23:27
experts is activated for a given token.
23:30
But if you remember our language model
23:32
architecture has multiple such
23:34
transformer blocks and there are experts
23:36
in each of these transformer block. So I
23:39
just want to show you the journey which
23:41
is traveled by an input sequence through
23:43
multiple such transformer blocks. Right?
23:46
So let's say the input sequence is what
23:49
is 1 + 1 followed by a question mark.
23:52
Okay, that's my input sequence right now
23:55
and let's say I'm focusing on this one
23:58
right now as a token. So this token is
24:01
converted into a token embedding and it
24:04
will be if I'm looking at one
24:06
transformer block and if I have four
24:07
neural networks or four experts, it will
24:10
be routed to one of the experts. So if
24:12
the sparity factor is 1x4, all these
24:14
other experts will not be utilized. It
24:16
will be routed to this expert which
24:19
deals with numbers. So in this first
24:22
transformer block, this one will be
24:23
routed to this expert and then we get
24:26
some output. What I want to show here is
24:28
that there are different such
24:30
transformer blocks, right? There might
24:31
be 12 transformer blocks and there are
24:33
multiple experts in each transformer
24:35
block. So if this one is routed to a
24:38
particular expert in the first block, it
24:40
is completely possible that it's routed
24:42
to a different expert in the second
24:43
block. It'll be routed to a different
24:45
expert in the third block and right till
24:47
the 12th block, it will be routed to a
24:49
different
24:50
expert. So it does not necessarily mean
24:54
that if if one is routed to expert
24:58
number one in the first transformer
24:59
block, it will only be routed to expert
25:01
number one in all the blocks.
25:03
No experts in different transformer
25:06
blocks might be specialized on different
25:08
tasks. So if expert one handles numbers
25:11
in the first transformer block, expert
25:13
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







