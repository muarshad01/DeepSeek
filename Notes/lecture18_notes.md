#### Mixture of Experts

***

since they are very focused on LLMs. I'm
5:36
not going to give you a historical
5:38
context of mixture of experts because
5:40
there is a whole uh literature on that.
5:43
I'm directly going to start with
5:45
language model.
5:46
So if you look at the transformer block,
5:49
the transformer block looks something
5:50
like this. We have uh we have an input.
5:54
So let me play this GIF once for your
5:56
understanding. We have an input input
5:59
block where we have tokenization, token
6:02
embeddings, positional embeddings. We
6:04
have the transformer block where all the
6:06
magic really happens. Here we have the
6:08
layer normalization, multi head
6:10
attention dropout, another layer
6:12
normalization, feed forward neural
6:14
network dropout and then we have an
6:16
output layer where the input embeddings
6:19
are converted into the logits matrix and
6:21
then we predict the next
6:23
token. Mixture of experts is an
6:26
innovation which is especially concerned
6:28
with one block among this entire
6:31
architecture and that's this feed
6:33
forward neural network. So if you
6:36
carefully look at this feed forward
6:37
neural network, it kind of looks
6:39
something like
6:40
this where if this is the input
6:44
embedding dimension, it projects it the
6:46
feed forward neural network projects the
6:48
input embedding dimension into a hidden
6:50
layer whose dimension is four times the
6:53
embedding dimension and then it projects
6:56
it back into the original input
6:58
dimension. So I like to call this neural
7:01
network the expansion contraction neural
7:03
network. Here there is an expansion when
7:06
we go from the initial layer to the
7:08
hidden layer and then finally there is a
7:10
contraction from the hidden layer back
7:13
to the original
7:15
dimension. So when we use a neural
7:17
network like this it retains the
7:18
original dimension but the input
7:22
embedding goes into an expansion and a
7:24
contraction. The reason we use a feed
7:26
forward neural network here is because
7:28
it allows the language model to explore
7:30
a much richer space. It increases the
7:33
dimensions the number of dimensions by a
7:35
huge amount. So it is one of the
7:37
critical components of the transformer
7:39
architecture and experiments have shown
7:42
that if this component is removed the
7:45
language models don't perform as well.
7:47
So the feed forward neural network is
7:49
one of the key components of the
7:51
language modeling
7:52
architecture. But one thing to note is
7:55
the number of parameters which the feed
7:56
forward neural network actually takes.
7:58
Right? So let us try to estimate this.
8:01
If you have a transformer
8:03
block, one transformer block let's say
8:06
and the input embedding dimension is
8:08
768. So how many parameters will be
8:10
there in one such feed forward neural
8:13
network. So 768 in the let's see in the
8:16
expansion layer we have 768 inputs and 4
8:19
into 768 hidden layer dimension. So the
8:22
number of weights will be 768 into 4 *
8:27
768. Those are the number of parameters
8:29
in the expansion layer and the number of
8:32
parameters in the contraction layer are
8:34
similar. So the number of parameters in
8:36
the contraction layer are again
8:38
768 multiplied by 4 *
8:41
768. Correct? So if you do this
8:44
calculation, let's do this. So 768 * 4 *
8:49
768 that's going to be equal to this.
8:51
And if I multiply this by 2, that's
8:53
going to be equal to this. So this is
8:55
actually equal to 4 million right and if
8:59
you multiply this by 12 so if the number
9:01
of transformer blocks are 12 that is
9:04
around 56 million
9:07
parameters. Um so this is for one
9:09
transformer block and that's around 4
9:12
million parameters. So if you have 12
9:15
such transformer blocks you get around
9:17
48 to 50 million parameters which are
9:19
present in the feed forward neural
9:20
network. The reason I'm showing this to
9:23
you is because the feed forward neural
9:24
network has a lot of parameters and what
9:27
this does is that this affects the
9:29
training time of the language model as a
9:31
whole. This increases the training time
9:34
of the language model and it also
9:35
increases the inference time of the
9:37
language model. Mixture of experts
9:41
optimizes this by reducing the
9:43
pre-training time and it also actually
9:45
reduces the inference time. And now
9:48
let's see how mixture of experts does
9:50
this.
9:52
So first uh what I have written here is
9:54
that we call this neural network the
9:57
compression expansion or the expansion
9:59
contraction neural network. Um what
10:03
mixture of experts does is that we have
10:05
multiple such neural networks which are
10:07
called as experts which are present in
10:09
the transformer block. So instead of
10:12
having only one neural network over here
10:14
we have multiple neural networks.
10:18
uh in this figure what I have shown here
10:20
is that there are four such neural
10:22
networks and each neural network which
10:24
is now present in the transformer block
10:26
it is called as an expert. So remember
10:29
here I showing with respect to one such
10:31
transformer block right in one
10:34
transformer block instead of having just
10:35
one neural network we now have 1 2 3 and
10:39
four neural networks which are present
10:42
and such four neural networks are
10:44
present in all the transformer blocks.
10:46
If you have 12 transformer blocks, we
10:48
have four neural networks in all the 12
10:50
transformer blocks. Each neural network
10:52
here is called as an
10:54
expert. Now you might be thinking what
10:57
is the main advantage of this right?
10:59
Because it seems that we are adding more
11:02
things. So how does it really help with
11:05
respect to the pre-training and the
11:06
inference time. Uh so the main reason or
11:10
what these experts actually do is that
11:12
what's the need to add multiple experts?
11:14
So adding these experts actually allows
11:16
the model to be pre-trained with far
11:18
less compute as compared to a dense
11:21
model which is without
11:22
experts. And adding such multiple
11:25
experts also allows for a much faster
11:28
inference as compared to a dense model
11:30
without experts. Why is this possible?
11:33
How come adding more neural networks
11:36
reduces the pre-training compute time
11:38
and also the inference compute time?
11:41
Again let me go through what's written
11:43
here. Adding multiple experts allows my
11:45
pre-training to be done with far less
11:47
compute and adding multiple expert
11:50
allows my inference also to be done much
11:53
faster. How is this done? So the main
11:56
hint is that you have this word here,
11:58
right? And that covers everything that's
12:01
dense. If you don't have a mixture of
12:03
experts, then the neural network is
12:05
dense. Meaning all the parameters, all
12:07
the weights are completely activated
12:09
here, right? Nothing is set to zero. But
12:11
the whole idea of mixture of experts is
12:13
something which is called as
12:16
sparsity. So in a dense model every
12:19
input token passes through all the
12:21
parameters that is all layers and
12:22
neurons. So whichever input token which
12:25
I have here it passes through all of
12:27
these parameters all four million
12:28
parameters in in in each transformer
12:31
block. Every input token passes through
12:33
all these parameters. So I have to do
12:35
all those computations.
12:36
But in a mixture of
12:38
experts, only a small subset of these
12:41
are activated for any given input. So if
12:44
there are 64 experts, I'll have only two
12:47
of these experts activated at one time.
12:50
And if you have 64, you can even make
12:52
sure that the dimensions of each are
12:54
smaller. So if the dimensions of each 64
12:57
experts are smaller than 56 768 and I
13:00
have only two experts which are
13:02
activated for every token that will
13:04
reduce my amount of computations by huge
13:06
amount by a huge amount right because
13:08
now if I have 64 such
13:12
experts and if the size of each is let's
13:15
say 128 the dimension and I have only
13:18
two such experts which are activated so
13:20
that's 128 into 2 which is 256 this is
13:23
three times smaller than 768. So the
13:27
whole idea is this idea of
13:30
sparity. The whole idea is that of
13:33
sparity which means that whenever I
13:35
encounter a new token, it will not be
13:38
passed through all of these experts. It
13:40
will be it will be handled by only a
13:42
subset of these experts and that is
13:45
decided by something called sparsity
13:47
factor. Here I have shown only four
13:49
experts, right? But actual models have
13:51
maybe around 64 experts. So if you have
13:54
64, every token will only be passed to
13:57
two of them. Let's say all others will
13:59
be inactive during that time. What this
14:02
does is that every expert is specialized
14:05
in some sort of computation. And now I
14:09
let's say one token is passed. I don't
14:10
need to activate all the other tokens,
14:12
all the other experts. I only activate
14:14
those experts which are specialized for
14:17
that particular computation based on the
14:19
token which is given. So I reduce my
14:22
pre-training time. I reduce my inference
14:24
time as well by this trick of the
14:27
sparity. So this if if you can see this
14:30
particular figure this illustrates the
14:32
idea of sparity. This figure and some
14:35
other figures which I'm going to show
14:37
you now are taken from this blog by uh
14:40
Martin Gurand. This is one of the best
14:43
blog for understanding mixture of
14:44
experts. I'll also attach the link to
14:46
this in the comments. So this is a
14:48
figure borrowed from that block and here
14:51
you can clearly see that let's say these
14:53
are my input tokens right and I'm
14:56
looking at one transformer block where
14:58
there are four tokens what I'm showing
15:00
here is that let's say I'm looking at
15:02
this particular token for
15:04
now you see that expert one expert 2 and
15:07
expert 3 are not activated for this
15:09
particular token only expert 4 is
15:12
activated this means that there is spar
15:15
sparity factor of 1x4
15:17
which is 25. So for every token in this
15:20
input sequence, it will be routed to
15:22
only one of the four experts and then we
15:25
get the output that reduces my
15:28
pre-training or the uh inference uh
15:32
time. So this whole idea of sparity and
15:35
mixture of experts was not novel. It was
15:37
introduced in 1992 paper actually 1991
15:41
paper which I showed you at the start of
15:43
this lecture. But they used it for a
15:46
very new uh task. What they showed was
15:49
that they used this idea of picture of
15:51
experts
15:53
uh for the task of war discrimination.
15:55
So they used this wel discrimination
15:57
task and they showed that this can be
16:00
achieved not by one neural network but
16:02
by multiple neural networks and by
16:04
selecting the neural networks in a
16:06
sparse manner. So they have a figure
16:09
here. You see this is a really important
16:12
figure because what we are going to see
16:14
eventually in the mathematics of mixture
16:16
of experts is the same thing. So we have
16:20
borrowed this concept which was
16:22
innovated around 30 to 35 years back
16:24
which again highlights the power of
16:26
doing research and how research
16:28
accumulates across multiple years and
16:30
helps us. So we really stand on the
16:32
shoulder of
16:34
giants right. So they have used mixture
16:36
of experts uh for wavel recognition and
16:40
for wavel identification and they have
16:42
clearly seen that based on the
16:43
particular
16:46
wavel is actually
16:49
activated. So they show that
16:52
um based on the input there is a
16:55
specific expert which is routed to that
16:58
input. All the experts are not used for
17:00
every input and which is the whole idea
17:02
of sparity and they tested it with four
17:05
experts, eight experts
17:07
etc. And if you take a look at the
17:09
abstract, let's read out the abstract.
17:11
We present a new supervised learning
17:13
procedure for systems composed of many
17:15
separate networks each of which learns
17:18
to handle a subset of the complete set
17:20
of training cases. Uh which means that
17:22
each expert is specialized in some sort
17:25
of a
17:26
task. it. Therefore, we demonstrate that
17:29
the learning procedure divides up a
17:31
world discrimination task into
17:32
appropriate subtasks each of which can
17:35
be solved by a very simple expert
17:37
network. So, let's say if you have a
17:39
task which is a complex task it is
17:41
divided into subtask and each of these
17:43
subtask is performed by a separate
17:45
neural network right. So not we don't
17:48
need a one giant network. It's split
17:50
into multiple smaller network and each
17:52
of these expert networks does a specific
17:56
task.
17:58
Um there is a thing called a selector or
18:01
a gating network which basically selects
18:03
which expert to use for the given token.
18:06
We'll come to the mathematics later. For
18:08
now just remember that for every token
18:10
we don't have all the experts which are
18:12
activated but only a selected set of
18:14
experts which is the whole idea of
18:17
sparity. Out of this lecture if there is
18:19
one takeaway you should have is the
18:21
concept of
18:23
sparity.
18:25
Um all right now what I want to show you
18:28
is that I want to show you what do these
18:30
experts really learn. So I've been
18:32
telling you that the experts specialize
18:34
in certain type of tasks. Right. So you
18:36
might be curious for a language modeling
18:39
uh for a language modeling example what
18:41
do these experts really learn. So if you
18:44
take a look at this paper this is called
18:46
STM OE
18:48
um designing stable and transferable
18:51
sparse expert models and uh this was
18:55
published in 2022. It's one of the most
18:57
famous mixture of experts paper. And if
18:59
you scroll down below
19:02
uh if you reach towards the end here
19:05
they have a table number 13 where they
19:08
actually show what the different experts
19:11
are learning. So here the layer
19:13
corresponds to the transformer block. So
19:15
if you look at layer number two you see
19:17
that the experts in this layer are
19:20
learning the punctuation. So whenever
19:22
there is a punctuation uh in the input
19:25
sequence that is routed to this partic
19:26
these particular experts who are trained
19:28
to understand punctuation. So this
19:31
column is routed tokens which means that
19:33
whenever we have a punctuation it will
19:35
be routed to these experts in layer two.
19:38
Similarly if you look at conjunctions
19:40
and articles the the and if etc that's
19:45
routed to experts in layer number three
19:47
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


