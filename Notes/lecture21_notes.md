my name is Dr. Raj Dandkar. I graduated with a PhD in machine learning from MIT in 2022 and
0:08
I'm the creator of the build deepseek from scratch series. Before we get started, I want to introduce all of you
0:15
to our sponsor and our partner for this series invido AI. All of you know how
0:20
much we value foundational content building AI models from the nuts and bolts. In Nvidia AI follows a very
0:27
similar principle and philosophy to that of us. Let me show you how. So here's
0:33
the website of Invido AI. With a small engineering team, they have built an
0:38
incredible product in which you can create highquality AI videos from just
0:43
text prompts. So as you can see here, I've mentioned a text prompt. Create a
0:49
hyper realistic video commercial of a premium luxury watch and make it cinematic. With that I click on generate
0:56
a video. Within some time I'm presented with this incredible video which is
1:02
highly realistic. What fascinates me about this video is its attention to detail. Look
1:08
at this. The quality and the texture is just incredible. And all of this has been created from a single text
1:15
prompt. That's the power of Invido's product. The backbone behind the awesome
1:21
video which you just saw is Invido AI's video creation pipeline in which they
1:26
are rethinking video generation and editing from the first principles to experiment and tinker with foundational
1:33
models. They have one of the largest clusters of H100s and H200s in India and
1:38
are also experimenting with B200s. Nvidia AI is the fastest growing
1:44
AI startup in India building for the world and that's why I resonate with them. so much. The good news is that
1:51
they have multiple job openings at the moment. You can join their amazing team. I'm posting more details in the
1:57
description [Music]
2:02
below. Hello everyone and welcome to this lecture in the build deepseek from
2:08
scratch series. Today we are going to understand the main innovations which
2:15
deepseek implemented in the mixture of experts architecture. Until now we have seen
2:21
three lectures in mixture of experts. In the first lecture, we looked at the mixture of experts
2:28
um introduction and how we replaced the feed forward neural network in the
2:34
traditional transformer architecture with a bunch of neural networks which were called as
2:40
experts. That's that was the first thing which we saw in the first lecture on mixture of experts. Then we also looked
2:47
at the intuition behind why mixture of experts really works. mixture of experts
2:53
drastically improves the pre-training efficiency and also the inference efficiency and one main reason for this
3:00
is the concept of sparsity. We saw that when we look at
3:07
every individual token, it's not really routed to all the experts, but it's routed only to a fraction or a subset of
3:15
the experts. And that's really called as a sparse model where if you have four
3:20
experts, maybe only one expert is activated for every token. So sparity is
3:26
one of the main ideas behind which mixture of experts really operates.
3:31
Then in the second lecture we saw the step-by-step procedure through which mixture of experts is implemented. We
3:38
first take the input matrix and we multiply it with the routing matrix. That gives us the expert selector
3:45
matrix. Then we decide how many experts need to be routed to each for each token
3:51
and that's called as top K. So if top K is equal to two, we select two experts
3:57
for every token. The way these experts are selected are based on the values of the expert selector matrix. So if you
4:05
look at every row here that corresponds to the experts to which every token is routed. So the first token will be
4:12
routed to these two tokens. The second token will be routed sorry the first token will be routed to these two
4:17
experts. Second token will be routed to these two experts etc.
4:22
So we take the expert matrix and then we apply soft max and that gives us the
4:28
expert selector weight matrix where if you look at every row it adds up to one.
4:34
So the value each value corresponds to a probability with which that token is routed to that particular expert.
4:42
Uh so that is how what we ultimately do is that we use these probabilities as uh weight as
4:50
weights and then based on any given token we first look at which experts is
4:56
it routed to and then we use these waiting factors we use these waiting factors or these weight factors to add
5:05
the three outputs. So we have three experts. So there are three expert outputs since there are three neural
5:11
networks. And then for every token based on the experts it's routed to and based on the waiting factor we use a weighted
5:18
summation and get the final output vector for every token. All these output
5:24
vectors are then aggregated together and then we get the resultant output for the input. So if the input is
5:32
4a 8 the output is also 4a 8 but now it's a cumulative sum of three neural
5:38
network output instead of just one neural network output. That's the main idea of mixture of experts which we saw
5:45
until step number seven. In the previous lecture we looked at balancing
5:50
techniques. So we want to ensure that ultimately when we look at all the
5:56
experts we want all experts to be operational. We don't want that one expert hog the limelight and one expert
6:04
received all the tokens. That's not what we want. We actually want all the experts to participate. So if you look
6:11
at every expert towards the end, every expert should on an average receive equal number of
6:17
tokens. That's called as a balanced um balanced mixture of experts model. And
6:23
to do that we saw couple of different things. First we saw something which is called as the auxiliary loss. where we
6:30
quantified the expert importance. So we found the importance of every expert based on the probability summation of
6:36
tokens routed to that expert and then based on the expert importance we
6:42
obtained the coefficient of variation which is the standard deviation divided by the mean and then we added this
6:48
auxiliary loss factor to the overall loss. So higher the auxiliary loss it means higher the coefficient of
6:54
variation which is not good because then one expert will be assigned more importance compared to other
7:00
experts. Then we saw that this term was not enough uh to make sure
7:06
that tokens are routed uniformly to different experts. So then we introduce
7:12
some a term which is called as load balancing and the load balancing loss essentially was defined by this scaling
7:18
factor multiplied by the number of experts and then for every experts we have two terms FI and
7:25
PI. This FI denotes the fraction of tokens sent to a particular expert I and
7:32
PI denotes the fraction of probability which is allocated to that expert I. Now
7:38
minimizing this loss, it turns out that minimizing this loss gets us a balance
7:43
in terms of the fact that if a token if a expert has more importance more number
7:49
of tokens will be routed to it. If a expert has less importance less number of tokens will be routed to it. So there
7:56
will be that proportionality between FI and BI. So
8:02
what minimizing this loss what it does is that um it allows experts with higher
8:09
importance to handle proportionately more tokens and it allows experts with lower importance to handle
8:15
proportionately fewer tokens. So by minimizing this second loss which
8:21
is called as the load balancing loss, we mathematically enforce the model to distribute tokens proportionally to how
8:28
much the expert is valued or trusted and that leads to a more balanced use of the mixture of experts
8:34
architecture. We also saw something called as the capacity factor which limits the number of tokens every expert
8:40
can handle. Okay. So today the key thing which we are going to uh remember is the
8:48
second loss which is called as the load balancing loss and there are
8:53
these terminologies fib. If you don't remember this it's fine but just remember that the load balancing loss is
9:00
added as a extra term to the total pre-training language loss which we
9:05
already have when predicting the next token. Today we are going to learn about the innovations which deepseek
9:12
implemented to deal with these losses. So Deepseek actually what it did was the
9:18
the whole mixture of experts literature was happening and like Deepseek did with
9:25
so many other innovations which they have. They took this mixture of experts innovation and then they built on top of
9:32
it which means they took all these loss terms which we have been seeing. They modified these loss terms to make it
9:37
more efficient. And today we are going to see three main innovations which DeepS seek implemented. The first is
9:44
something which is called as auxiliary loss free load balancing. The second is
9:49
called as shared experts and the third
9:54
innovation is called as fine grain expert segmentation. We are going to look at all of these three innovations
10:00
today. But first let me give you a bit of history before looking at these innovations. The first major paper which
10:08
implemented many of these innovations was deepseek ME which is deepseek
10:13
mixture of experts that came out in January 2024. This paper implemented two
10:18
innovations which I discussed right now. Innovation number two which is shared experts and innovation number three
10:24
which is fine grained expert segmentation. These two innovations were implemented by Deepseek in their version two paper
10:31
which came out in June 2024. And in January of 2025, there was
10:37
the version three paper which led to the whole Deepseek revolution. Here those two innovations were still implemented.
10:43
The fine grain expert segmentation and shared experts was still implemented. But there was one more innovation which
10:50
was implemented specifically in this paper and that innovation which and that
10:55
innovation is called auxiliary loss free loss free load balancing. This was not implemented in version two but this was
11:02
implemented in version three. So we are going to start looking at this innovation initially and then we are
11:08
going to look at innovation 2 and innovation 3. So let's start looking at the first deepseek innovation in the
11:14
mixture of expert space and that's auxiliary loss free lossfree load
11:20
balancing. So let's start with the first deepseek innovation auxiliary loss free
11:27
load balancing. So this innovation starts from the step number nine. In step
11:34
number nine, we saw this load balancing loss, right? Which had this fraction of
11:42
tokens sent to expert I fi and the probability allocated to every expert pi. This was the load balancing
11:49
loss. Now this loss was successful because it helps to
11:56
maintain load balance in expert models. But the main issue with this loss is that this loss is actually added to the
12:03
training loss. So let's say I have my training loss for the next token prediction and I use this scaling factor
12:11
times the number of experts time sigma fi pi that's my load balancing loss
12:17
correct the main issue is that this loss actually interferes with my training
12:22
loss because if you look at both of these losses they lead to they denote fundamentally different things right my
12:29
training loss denotes how well I'm doing on the next token prediction Whereas this loss encapsulates the
12:37
information of balance between my different experts and whether tokens are routed uniformly to experts etc. So both
12:45
of these losses mean fundamentally different things and adding them both together and then taking the gradients
12:51
and back propagation is very inefficient. So use of this loss helps
12:57
maintain load balance in expert models. That's the advantage. But this loss also acts as a regularization term that
13:04
interferes with language modeling. For example, if the scaling factor, let's
13:10
say if the scaling factor is very small, then this term is almost negligible. Right? So if the scaling factor is small
13:16
or absent, uh this term is almost negligible and we won't have good expert utilization. So
13:22
then the mixture of experts model will be almost useless. Whereas if lambda is
13:28
very high, if lambda is very high then we'll it what it means is that it degrades the model performance.
13:36
So if lambda is very high, we'll give more importance to this loss compared to this first loss. So we'll not have good
13:42
performance on the next token prediction task if lambda is high. So if lambda is low, we still have a problem because
13:49
then the mixture of experts model won't be balanced. And if lambda is high, we still have a problem because then my uh
13:56
next token prediction loss will be very low. So the main issue is that balancing
14:05
it's difficult to balance experts effectively without compromising training quality. That's the main thing.
14:11
If we go to balance the experts, the training quality will be reduced. So if this quality increases which means if my
14:18
mixture of experts model is to be more balanced lambda needs to be higher and then that will affect my training
14:24
quality and that's a big trade-off over here which means that adding this term over here is reducing my training
14:31
quality and deepseek noticed that problem and this is where they
14:36
implemented something which is called as lossfree balancing lossree balancing which meant
14:43
that they completely completely got rid of this term. They got they completely got rid of the second term and they just
14:51
had one term in the training loss which is the next token prediction
14:56
task. So then you might be thinking that if deepseek completely got rid of this
15:03
second term then how did they implement load balancing? How did we make sure
15:08
that every expert handles relatively equal amount of
15:14
tokens and how do we get a balanced mixture of experts model? So they implemented a different
15:22
technique. They implemented a different method that enforces load balance
15:28
without using additional gradients from this loss. So they did not have this loss at all. So we did not have to take
15:34
gradients during back propagation. So let's see the technique which they used to implement load balance. So D6 still
15:40
implemented load balance without the loss
15:47
term and I'll show you how they implemented this. This thing which they
15:52
implemented is shown in a very small section. Uh this is a section which is
15:58
called as auxiliary loss free load balancing and it's equation number 16. In just two paragraphs they explain this
16:05
innovation which they have but I'm going to try to explain to you in as much detail as possible. Okay. So let's say
16:13
we have the expert selector weight matrix which I showed you before. Right? So what this expert selector weight
16:19
matrix tells me is that if my first token comes in, this is my first token. So let me first of all reduce the
16:25
thickness a bit and change the color. If my first token comes in, it's routed
16:30
to expert 2 and expert 3 with probabilities 6 and 04. If my second token comes in, it's routed to expert
16:37
one and expert three. If my third token comes in, it's routed to expert two and expert three. If my fourth token comes
16:44
in, it's routed to expert number one and expert number three. Okay. So the
16:50
first in this technique implemented by Deepseek load balance without loss.
16:56
First thing we can do is we have to find the average load. We have to find the average load
17:04
per expert. Or let me call it average token
17:09
load per expert. What this means is what are the
17:14
average number of tokens which are routed to every expert. So first for that we have to see the tok total number
17:21
of tokens which are routed. Right? So first token uh total number of tokens
17:26
routed. So first expert has how many tokens routed? One and two. Second token
17:32
second expert has how many tokens routed? 1 and two which is two. And third expert has how many tokens which
17:38
are routed to it? 1 2 3 and four. So four tokens. So overall the total number
17:43
of tokens which are routed to these three experts is 2 + 2 + 4 that is eight
17:49
tokens. So eight tokens are routed to these three experts. So what is the average number of tokens which are
17:55
routed per expert? That is just 8 divided by 3 which is equal to 2.67. So the average token load per
18:02
expert is 2.67. Okay. Now based on the average load per expert we can find out whether
18:10
a given expert is overloaded or underload. So for example if you look at the first expert right how many tokens
18:17
are routed to it? two tokens are routed to it and my average load per expert is 2.67. So this expert is actually
18:25
underloaded underloaded by an amount of 2.67 which is the average token load per
18:30
expert minus 2 which is 67. This means that it actually has a
18:35
capability of handling 2.67 tokens but only two tokens are routed to it which
18:41
means it's underloaded by an amount of 67. Similarly, if you take a look at the
18:47
second expert now, only two tokens are routed to it. So, the second expert is also underloaded by an amount of 2.67
18:54
minus 2, which is 67. And if you look at the third expert, however, the third
18:59
expert has four tokens routed to it. So, the third expert is actually overloaded
19:05
because the number of tokens here are higher than the average load per expert, which is 2.67.
19:10
And what's the amount by which the third expert is overloaded? That amount is 2.67 - 4 which is -
19:18
1.33. So to get the amount of overload or underload, we just take the average token load per expert and subtract the
19:26
number of tokens which are routed to that expert. So the third expert is overloaded right now and my first two
19:32
experts are underloaded. So as seen in the figure above we see
19:38
that experts one and two are underloaded and expert 3 is overloaded. Then what we do is that we find the load violation
19:45
and we have already found out the load violations in this figure. Right? The load violations is
19:52
just the difference between the number of tokens routed and the average token
19:57
per expert. So the load violation for expert 1 and 2 is 67 which is positive
20:03
and load violation for expert 3 is minus 1.33. So for expert one load violation
20:10
is expert one and two it's 67 which is positive and for expert three the load
20:18
violation is minus 1.33 which is negative. Okay. Now once we capture the load violation
20:25
error for the different experts then we introduce a terminology which is called as
20:30
bias. Now this bias is a term which is introduced as zero for every
20:36
expert and then it's updated and then it's updated per iteration for every
20:43
expert. So here is how the bias term is updated. Right? So bias term is equal to
20:49
the bias term plus u which is a predefined constant multiplied with the
20:54
sign of the loadation error which means that if it's if the loadation error is
21:00
positive then this will be b i is equal to b i + u and if the loadation error is
21:06
negative it will be b i is equal to b i minus u. Now we have already seen when is the loadation error
21:13
positive when it's underloaded right. So expert one and two were underloaded here and the loadation was
21:20
positive. So for experts one and two the bias will be increased. For experts one
21:26
and two the loadation error will be positive. Experts one and two and for
21:32
expert number three bias will be decreased because this expert is already
21:39
overloaded. Uh so the above formula ensures that if an expert has a heavy load, we will reduce its bias otherwise
21:46
we'll increase it. For example, expert number three, it has a heavy load, right? It's overloaded. Its load
21:52
violation is negative. So we are going to reduce its bias. Whereas experts one
21:58
and two, they have a load violation of positive. They are underloaded. So we
22:03
are going to increase the bias of experts one and expert two. Then what do we do with this bias term? Once we have
22:10
this bias term so first of all I've shown this in visual format for experts
22:16
one and two these are underloaded so we increase the bias and for expert three
22:21
this is overloaded so we are going to reduce its bias all right now where is
22:28
this increase and decrease actually implemented so remember after we
22:35
multiply the input matrix with the routing matrix we get a matrix like this. So if you take a look at our
22:41
earlier steps which we implemented in the mixture of experts model, we multiply the input matrix with the
22:47
routing matrix and we get this expert selector matrix, right? This expert selector matrix is where this bias term
22:54
is actually added. So you have this expert selector matrix, right? And then we adjust this
23:01
with the bias term, which means that experts one and experts two, we now add
23:06
a bias term. See because these experts are experts one and experts two are
23:12
underloaded. So we are increasing the bias term. So for these experts we add a
23:17
plus b term. We add a plus bc to all the numbers which were previously there we add a plus b. And to expert number three
23:25
we add a negative term which means we subtract we subtract from
23:31
this. So my final numbers will become will increase for experts one and two
23:36
and will decrease for expert number three. So since experts one and two are
23:41
underloaded adding the bias term will make sure that the value of expert one and two will increase. This will
23:49
increase the probability of these experts being chosen by the router. That's the most important thing. So
23:54
experts one and experts two are
24:00
underloaded are underloaded. So we add the
24:07
bias and we increase the probability of being chosen by the
24:19
router. Whereas for expert number three it's overloaded. So it goes in the other
24:24
direction. So expert number three is overloaded. So we reduce the
24:33
bias and we decrease we decrease the probability of this expert being chosen
24:40
by the
24:45
router. So expert 3 is overloaded. So subtracting the bias term will make sure
24:50
that the values of expert 3 will decrease like this and this will
24:55
decrease the probability of this expert being chosen by the router. So you see what is happening
25:01
here. The experts to which more number of tokens are routed like expert number
25:06
three we decrease the bias term from these values so that it gets less tokens
25:13
next time because the router won't select it as its probabilities are reduced.
25:20
Whereas experts one and two have less number of tokens routed to it. So they are underloaded. So we increase the bias
25:27
or we add the bias that increases the probability of more tokens being routed to expert one and expert two. So that's
25:34
how we are maintaining the balance. You see by dynamically adding or subtracting
25:39
the bias terms. We are eventually ensuring that all the experts relatively receive equal number of tokens. How are
25:47
we ensuring this? Because in an iteration if an expert suddenly receives high number of tokens we'll subtract the
25:53
bias term from it like we did from expert number three that will reduce the probability of the router routing
26:00
anything to that particular expert. Similarly during any particular iteration if some experts are
26:05
underloaded we add the bias term to these experts and that increases the
26:11
probability of the tokens being routed to those experts.
26:16
So with this dynamic adjustment of the biases, DeepS seek achieved good expert load balance without actually
26:23
introducing any noisy gradients to the model. We now have no loss term. You see
26:28
the beauty of this approach. We don't have any loss term like we had over here. Right? This load balancing loss
26:35
term is now not there at all. And that's why it's called auxiliary lossree load
26:41
balancing. Which means that the load balancing which we have implemented is loss free. We have no loss term in this
26:47
load balancing at all. We are doing the entire load balancing through dynamic
26:52
adding and subtracting of the bias term. Uh and through this what DeepSync
27:00
did is they somehow always achieved the best of both worlds like they did with latent attention. Here the best of both
27:06
worlds they achieved is they also got a good uh training loss because now the
27:12
training loss is not interfered with the second loss. So they maintained the training loss which means next token
27:18
prediction was not compromised while at the same time they maintained load balancing. They made sure that all
27:25
experts receive relatively equal number of tokens without adding any loss term at all. And that was one main key
27:32
innovation where deepseek actually showed that auxiliary loss free load balancing achieves both better
27:39
performance and better load balance compared with traditional load balancing. Okay, please keep this result
27:47
in mind. Deepseek showed that auxiliary loss free load balancing achieves both
27:52
better performance and better load balance compared with traditional load balancing. And the first time um so they
28:00
actually proposed auxiliary loss in one of the earlier papers but uh the version
28:07
three was I believe the first version to implement this auxiliary loss free load balancing. I'm I don't know whether loss
28:15
I don't think lossree load balancing was implemented in version two but version 3 had a
28:22
section on loss free load balancing where this BI is the bias term which we introduced if you directly look at this
28:29
formula it will be very difficult to understand how it's actually implemented but that's why I broke it down into this
28:35
visual explanation so that all of you can understand how this bias term is actually implemented to implement load
28:42
balance uh so now when you read this section you will understand it much better by taking
28:50
into account this visual explanation in mind so until this stage in this lecture
28:56
we have understood the first innovation by deepseek in the mixture of experts and that's auxiliary loss preload
29:03
balancing now we move to the second innovation which is called as shared experts and then we'll also look at the
29:10
third innovation which is fine grained expert segment mentation. So let's take a look at these now. So we are going to
29:16
look at two innovations now and we are going to look at these two innovations in parallel. The first is shared experts
29:24
and the second is fine grained expert segmentation. The reason we are going to look at these two experts in par in
29:31
these two innovations in parallel or simultaneously is because deepseek first talked about these two innovations in
29:37
their mix deepseec paper um which came out in January 2020 2024. Here deepse
29:46
outlined two major problems with traditional mixture of experts uh architecture and then they proposed
29:53
these innovations as possible solutions to these problems. The first problem which deepseek talked about is something
30:00
which is called as knowledge hybridity. So they said that existing mixture of expert practices often employ limited
30:07
number of experts. So let's write this down the problems which they actually uh
30:12
suggested. The first problem is that of knowledge hybridity.
30:19
So what they claimed was the existing mixture of experts model had limited number of experts
30:28
um had limited number of experts and thus token assigned to a specific experts will be
30:34
likely token assigned to a specific expert will be likely to cover diverse knowledge. Consequently, the designated
30:41
expert will intend to assemble vastly different types of knowledge in its parameters which are hard to be utilized
30:48
simultaneously. So what they mentioned was models had limited number of experts. So let's say experts were of
30:55
the order of 8 to 16, eight experts, right? So if I only have eight experts
31:00
and I have this huge amount of data, it means that every expert will need to have knowledge about probably so many
31:06
different things which means that I won't have specialized
31:12
experts. It would be like my expert is each expert is trying to do all different things and so every expert
31:19
does not become great or specialized in one particular endeavor or in one particular task. my expert tries to have
31:27
all the knowledge in various different fields
31:32
um and as a result it might be harder to utilize that knowledge. That's the first issue which they mentioned which is
31:38
called as knowledge hybridity and the main issue here was limited number of experts. The second
31:45
issue which they mentioned is knowledge
31:50
redundancy and I believe this was an even bigger problem because
31:55
here what they actually said was tokens assigned to different experts may
32:02
require common knowledge. As a result multiple experts may converge in acquiring shared knowledge in their
32:08
respective parameters leading to redundancy in expert parameters. What this means is
32:14
that let's say again I have this limited number of experts and uh let's say this
32:20
expert receives a certain set of tokens and that requires this experts to this expert to have general
32:27
knowledge and let's say this expert and this expert also receives certain tokens which require both of these experts also
32:34
to have general knowledge. It actually means that all of these
32:41
three experts these three experts are now specializing in the same type of information or same type of knowledge
32:47
which means that the knowledge this expert has it's that same knowledge is also acquired by these two experts and
32:54
that is called as knowledge redundancy.
33:02
knowledge redundancy. Um what this means is that
33:10
this hinders the expert specialization which means that both of these issues which we have now discussed u the first
33:17
issue being that of knowledge hybridity where every expert assembles knowledge from different fields and second is
33:23
knowledge redundancy where again many experts might have the same knowledge.
33:28
Both of these because of both of these it's very
33:34
difficult to have specialized experts. By specialized experts we mean
33:39
experts which have specialized knowledge about particular task and whenever token comes in it will be routed for that task
33:46
only to that expert. So what deep set out to do is that it wanted to create
33:51
this super specialized experts. It wanted to create the super specialized
33:58
experts which solved both of these problems. They wanted to solve the knowledge hybridity problem. So they
34:04
wanted experts to have complete knowledge about a specific field rather than acquiring knowledge from
34:10
everywhere from various different fields. And second they wanted to solve the knowledge redundancy problem and
34:17
that's why they titled their um paper as towards expert specializ towards
34:23
ultimate expert towards ultimate expert specialization in mixture of experts model. So the way they set out to solve
34:30
both of these problems is the first problem of knowledge hybridity they claimed that instead of having limited
34:37
experts why don't we have a huge number of experts instead of having a few experts
34:44
since we have only few experts all of them have to learn many different things but if we have huge number of experts we
34:51
might have experts which are specialized in certain knowledge that's the first thing they implemented and that's the
34:57
innovation number three which is time ingrained expert segmentation. And the second thing which
35:02
they implemented is uh to prevent this knowledge redundancy. What if we have shared
35:08
experts? What if we have shared experts and this shared experts would
35:14
contain all of the common knowledge which is needed and along with the shared experts we again have the specialized experts. So for specialized
35:22
knowledge we have this other set of experts but we have the shared experts
35:27
who learn all of the redundant information and that's the first innovation or deepseek innovation two
35:34
which I have mentioned here that's called as shared experts. So first let me talk a bit about shared experts and
35:40
then I'll talk about expert segmentation. So in shared experts what
35:45
deepse did is that they divided their experts into two groups. They divided
35:51
their experts into routed experts which is commonly used in the traditional mixture of experts architecture and
35:57
shared experts. So the main difference is that if you look at the routed experts it's sparse which means that we
36:03
implement this top K routing and only certain number of experts are selected for every
36:10
token which is how it's done in traditional mixture of experts. But then they had this other group of experts
36:15
which is called as shared experts and which are always activated which means that any token all the tokens pass
36:22
through all of these experts which means if one token comes in it has to go through all of these experts. If another
36:28
token comes in it has to again go to all of these experts. So here there is no sparity which is implemented in this
36:34
shared experts which means all of the shared experts are always active whereas the routed experts are
36:42
selectively active as always right. So shared experts are experts that process
36:47
every token regardless of routing and routed experts are experts that handle token selectively. Why did they do this?
36:55
Because they wanted to reduce redundancy among experts. So by having shared experts the common
37:02
information and processing tasks can be centralized which means that all the common tasks like general knowledge etc
37:08
can be handled by these shared experts u and this allows routed experts to
37:15
focus on more specialized task. So if you think about it this shared experts handle the common
37:23
tasks and my routed experts then can handle specialized task. I don't need my router routed experts to have knowledge
37:30
of this same information and that leads to specialized experts which means an
37:36
expert over here might be completely specialized maybe in doing complex complex
37:43
arithmetic. So that just makes the architecture a lot more efficient since we have the super specialized experts
37:51
and that reduces the knowledge redundancy problem. The second problem of knowledge redundancy which was there
37:57
it can be completely reduced with the shared experts idea. So this is the main schematic
38:04
which you will also see with the shared experts and that is shown in the mixture of experts paper also. If you see this
38:10
is the idea of shared experts where what this this expert let's say is always
38:16
activated whereas these experts are sparsely activated. This green expert is always activated over here and this set
38:22
of experts are sparsely activated. This is the idea of shared experts and that's deepseek innovation
38:29
number two. Uh I think I already showed you. Yeah.
38:36
So now after this what is done is that the outputs from the shared experts and the outputs from the routed experts are
38:42
added together and that's how we get the resultant output. uh so summation of the output of both of
38:49
from both of these experts a weighted summation uh is done to get the final mixture of experts
38:56
output. So the outputs of the shared experts and routed experts are simply combined by adding together leading to
39:03
the final uh final output. Okay, that's the deepseek
39:11
innovation number two. And the deepseek innovation number three is fine grained expert segmentation. And as I've
39:18
mentioned here, this is the simplest to explain. The main idea is that if this is the conventional mixture of experts,
39:24
right, which has less number of experts in fine grained mixture of experts, we just divide every experts into multiple
39:30
smaller experts while maintaining the overall model capacity and computational cost. Which means that instead of having
39:37
four we now have maybe four into four where instead of having less number of
39:43
experts we have huge number of experts and the dimensionality is
39:48
maintained which means that um in fine grained expert segmentation each large
39:55
expert feed forward network is split into m smaller experts by reducing the hidden dimension by a factor of 1 by m.
40:02
Which means that now that I have more number of experts, the dimension of each expert is effectively reduced, right? Uh
40:10
because the ultimate dimension needs to remain the same. So computational cost does not change at all. Computational
40:16
cost does not increase because my number of model my model size remains the same. Ultimately I'm just having a huge number
40:23
of neural networks now instead of low number of neural networks and the dimension of each neural network here is
40:29
lesser than the let's say the original dimension and this idea of having more
40:35
number of experts is called as fine grained expert segmentation. U again why do we do this?
40:43
Because if the number of experts is small, each experts is for forced to learn a wide variety of knowledge type
40:49
which reduces its specialization. In fine grained expert segmentation,
40:56
uh we can have specialized experts because now that there are more experts, each experts can learn something new and
41:03
that solves the first issue which we started out with that's the knowledge hybridity. If you have limited experts,
41:09
every expert has to have a lot of information, right? But now if you have a huge number of experts maybe every
41:15
expert can be specialized in certain amount of or certain specific knowledge
41:20
and that leads to this super specialized experts. Um so even if you go to mixture
41:27
of experts original paper you'll see that in fine grained expert segmentation. So
41:33
see the first figure on the left hand side is conventional top two routing in mixture of experts. That's conventional
41:39
mixture of experts. When we go to the right first we add the fine grained expert segmentation where we have a huge
41:46
number of experts now and then as we go further to the right then we add a shared expert. So that's the first
41:52
innovation which we saw the shared experts and in the second figure we s we see this fine grand expert
41:59
segmentation. Uh one thing to note is that among all of these three the number of expert parameters and computational
42:06
costs actually remain the same. This this is because although we increase the number of experts the dimensions are
42:12
reduced appropriately so that the number of parameters the number of expert parameters actually remain the
42:19
same. Uh so if you see the mixture of experts paper and you'll see the main
42:25
innovations here have are the fine grain expert segmentation 3.1 and section 3.2
42:31
is the shared expert isolation and it contains everything which I've just shown to you right now. I believe in
42:37
version two and version three also uh these things were introduced. So if you look at version three for example
42:45
uh version three also mentions that they use fine grained experts and uh uh
42:50
shared shared experts but along with this they also introduce the loss preload balancing.
42:57
So if you directly start reading deepseek version 3 it will be very difficult to understand this whole
43:03
section because this entire section on mixture of experts they have compressed it in three paragraphs three or four
43:08
paragraphs. The first paragraph uh second third and four paragraph in
43:14
four paragraph they explain all of these innovations which we have seen right now. But to understand this innovations,
43:20
it was important for you all to understand how mixture of experts actually operates. And that's why we had
43:25
the hands-on mixture of experts lecture the first three lectures. Uh all right. So this is the
43:33
fine grain expert segmentation. And now now let me show you some results which deepse had in their paper to show how
43:40
these innovations actually led to improvement over traditional mixture of
43:45
experts. All right. So as I mentioned the last thing which I want to show you is some results which deepseek had in
43:52
their original MOE or mixture of experts paper and the first major major result
43:57
which I want to show you is over here. So if you look at the right hand side that's the deepseek mixture of experts
44:05
and uh here if you see the activated experts the total experts is one one
44:10
common expert which is the one shared expert and 63 routed experts and out of
44:15
those 63 only seven were activated. So the total number of expert parameters which were activated were only 24
44:23
billion and they compared this to another mixture of experts model which is called G-Shard. So if you look at
44:29
this Gstar 1.5 you'll see that they did not have group group experts because
44:34
that was the main innovation implemented by Deepc. So they just had 16 routed experts out of which two were activated
44:42
and the total number of expert parameters which they have is 2.83 83 billion and the activated expert
44:48
parameters which they had was around.35 billion right and they also
44:56
compared one more dense dense 16 model which did not have which only had 16
45:01
experts which did not have the fine grained expert segmentation and which did not also have any grouped um it was
45:09
a dense model which means all the tokens were routed to 16 experts all the tokens were routed to 16. So the number of
45:16
activated parameters were was 1.89 billion which means that the deepseek
45:22
number of activated expert parameters was around 1.5 times smaller than this
45:28
G-Shard and it was around six six times smaller than this dense 16. So this
45:35
dense there was no sparity implemented here. It was like there were 16 experts
45:40
and every top was routed to all 16. Now in spite of deepseek having so less
45:46
number of expert parameters you'll see that all of in all of the metrics deepseek com performed relatively at the
45:53
same level as these other models. So if you look at this accuracy metric deepsee had 54.8 8 which is equivalent to these
46:00
other models also although these other models had a huge number of parameters this dense 16 had six times more
46:07
parameters whereas in this metric let's say arc challenge accuracy deepseek mixture of experts actually was higher
46:15
than these other models that's incredible right so here this deepsee had fine
46:21
grained expert segmentation so they had much more number of experts u then they
46:27
also had shared experts But their number of expert activated parameters was way lower. So their computational cost was
46:34
way lower and still their accuracies was relatively comparable with the best mixture of experts model out
46:40
there. So here what they have mentioned is deepseek mixture of experts achieves comparable performance with a Gshar
46:48
model containing 1.5 times the expert parameter and computation. In addition, deepseek nearly approaches the
46:54
performance of a dense model with 16 times the number of parameters
47:00
um with 16 times the number of parameters which sets the upper bound fore models. So this table clearly
47:06
proves the effectiveness of the MOE architecture and its innovativeness such
47:12
as the fine grained expert segmentation, the shared experts compared to the traditional mixture of experts
47:18
architecture. Another figure here which I believe is very good is this one figure in which
47:24
you can see many things. First you can see let's say we don't have shared experts. So if you this blue line there
47:32
are different metrics here which have been plotted and the blue line is if we don't have shared experts and all the
47:39
other colors we do have shared experts. We have one shared expert for all the other colors. So now if you check the
47:46
performance on all of these matrix we see that the blue is much lower compared
47:51
to all the other colors right for the first metric blue is lower than all the other colors for the third metric it's
47:58
also lower for the fourth and fifth blue is definitely lower than all the other colors. This actually proves that having
48:04
a shared expert improves the performance of the mixture of expert model by
48:10
reducing the knowledge hybridity problem which we saw on the whiteboard. You remember we saw this knowledge hybridity
48:17
problem. This was only in theory up till now but deepseek actually proved this result by showing that if we have shared
48:24
experts it actually improves the model performance on a wide range of tasks.
48:30
The second thing we can get from this figure is that we can also see the
48:35
effect of fine grained expert segmentation. Right? So this yellow for example this yellow line does not have
48:42
fine grained expert segmentation. It only has 15 experts. Whereas the green and the the green line and the orange
48:50
line have fine grained expert segmentation. They have 31 and 63 routed experts. So you will consistently see
48:57
that the orange line performs the best among all. Right? The orange performs best across all metrics. This is because
49:03
the orange line has 63 routed experts. This again shows the importance of fine grained expert segmentation in solving
49:11
the knowledge redundancy problem. It shows it shows quantitatively that
49:16
having more experts uh having more
49:23
experts means I can have super specialized experts, right? So actually first I mentioned
49:30
that the shared experts solve the knowledge hybridity problem but actually
49:35
the shared experts solve the knowledge redundancy problem. Right? Because if you have shared experts, it means that
49:42
all the other experts don't need to assemble the same knowledge. The other experts can be specialized. So
49:49
this having the shared expert solves the knowledge redundancy problem. Whereas
49:55
having more number of experts, having fine grained expert segmentation solves the knowledge hybridity problem. So if
50:01
you have a limited number of experts, all experts are forced to learn everything. But if you have more number
50:06
of experts, every expert can be super specialized. So this solves the knowledge hybridity problem. Again
50:12
remember having shared experts solves the knowledge redundancy problem and having fine grained expert segmentation
50:19
solves the knowledge hybridity problem. So what they showed here quantitatively is that this orange line has fine
50:26
grained expert segmentation. Right? That solves the knowledge hybridity problem.
50:33
uh and this yellow line compared to the blue line we have shared experts. So
50:38
that solves the knowledge redundancy problem. So again to repeat it shared
50:43
expert solves the knowledge redundancy problem and fine grain expert segmentation solve the knowledge
50:49
hybridity problem. Again in this plot you can see across all the different metrics the
50:56
orange line is the highest and the orange line has shared expert plus fine grain expert segmentation. That proves
51:01
that having shared expert plus fine grain expert segmentation solves both
51:07
the knowledge the redundancy and knowledge hybridity problem and it leads to better performance across all of
51:13
these metrics which have been considered over here. They have several other results over
51:18
here which improve which show that how deepsek mixture of experts achieve super specialization using these
51:26
innovations. So that brings us to the end of today's lecture in which we have comprehensively seen the three main
51:32
innovations which power the deepseek mixture of experts. The first is uh
51:37
auxiliary loss free load balancing. The second is shared experts and the third
51:44
is finerained expert segmentation. These innovation two and three solve the problem of knowledge hybridity and
51:51
knowledge redundancy. Whereas the innovation number one which is the auxiliary loss preload balancing
51:57
reduces. We don't need to have this second loss term now which means that we can solely focus on reducing my training
52:04
loss and by dynamically adjusting this bias term. Now by dynamically adjusting
52:10
this bias term we can make sure that all the experts have equal amount of load.
52:16
So if some expert is underloaded we increase the bias. If some expert is overloaded we reduce the bias which
52:22
ultimately makes sure that the probability the router assigns to every expert is more or less uniform across
52:28
the experts. So what deepseek did is that they took they built upon something
52:34
which was already existing. They did not invent the mixture of experts architecture but they were smart enough
52:40
to notice the limitations in mixture of experts. Uh like there was a limitation in this loss term right when we added
52:46
this auxiliary loss term. When we added this uh this one this auxiliary loss
52:53
term uh there was a trade-off with the language performance. So they mitigated that trade-off by getting rid of this
52:59
term. Then they also added these two uh shared experts and
53:05
uh fine grain expert segmentation to solve the knowledge hybridity and knowledge redundancy problem. Everyone
53:11
had noticed these problems but deepseek was one of the first ones to take steps towards mitigated mitigating it and
53:18
that's the beauty of everything which they have done. They showed that a group of people if they came together even if
53:23
the resources are not that high we can still achieve incredible and innovative things and that's one of the main
53:30
inspiration why I'm making this series. Thanks a lot everyone. There are lots more lot more interesting
53:37
uh and advanced concepts to follow. So please stay tuned and make notes so that you'll understand and follow all. Thanks
53:44
everyone and I look forward to seeing you in the next lecture.
