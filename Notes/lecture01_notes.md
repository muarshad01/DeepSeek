#### How to build DeepSeek from Scratch?

* [DeepSeek R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)

***

* 5:00

| Year | Paper|
|---|---|
| Jan 2024 | [DeepSeek LLM: Scaling Open-Source Language Models with Longtermism](https://arxiv.org/abs/2401.02954)|
| Jan 2024 | [DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence](https://arxiv.org/abs/2401.14196) |
| Mar 2024 | [DeepSeek-VL: Towards Real-World Vision-Language Understanding](https://arxiv.org/abs/2403.05525)|
| Apr 2024 | [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)|
| Jun 2024 | [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) |
| Jun 2024 | [DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence](https://arxiv.org/abs/2406.11931)|
| Dec 2024 | [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)|
| Jan 2025 | [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)|

***

I was looking for there were R posts but
10:06
this was again not very useful or
10:08
impactful at
10:09
all um so then I decided that this time
10:13
I need to go my own route and I need to
10:15
do my own research I need to do my own
10:17
study and I need to build that first
10:19
video playlist which teaches deepik
10:21
fully from
10:22
scratch and the last aspect of this
10:26
playlist will be recreating something
10:28
like this mini R1
10:30
but to get to this stage we need to
10:31
explain every single thing in detail and
10:34
then I started diving into understanding
10:36
these papers I made huge amount of
10:38
detailed notes on these papers and all
10:40
the previous papers which I showed you I
10:43
already had the knowledge of standard
10:45
llm so that really helped me in my
10:47
research I made notes in all I spent
10:51
about I would say the last three weeks I
10:53
have spent about 10 to 12 hours every
10:56
day understanding everything about deep
10:58
seek making notes and ultimately I came
11:00
up with this plan of how to divide the
11:03
Deep siik architecture and how to teach
11:05
it to all of you so as you can see I've
11:07
divided it into chapters and I'm also
11:09
planning to write a book on it the first
11:12
is of course deep seek overview and
11:13
you'll see that the chapters or the
11:15
videos will be divided into two parts
11:17
architecture and modeling deep seek has
11:20
a huge number of Revolutions in their
11:22
architecture and these small things
11:24
added up to contribute to why deeps is
11:27
so good right they had some called
11:29
mixture of experts they had something
11:31
called multi-head latent attention they
11:34
implemented quantization they had
11:36
something called multi- token prediction
11:38
all of these things are not easy to
11:40
understand if someone just writes a
11:41
onepage report you cannot build a
11:44
mixture of experts model from scratch
11:46
multi-head latent attention is a topic
11:48
on which I have made notes of 50 pages
11:50
and only then I have been able to
11:52
understand it in fact I can show you
11:55
some of these notes which I made in Cana
11:57
if you go to the multi-head latent
11:59
attention part you'll see that these are
12:02
the kind of notes which I have made for
12:03
every single aspect of this so this is
12:06
what understanding from scratch means
12:08
right going down to the matrix
12:09
multiplication details and I'll be
12:11
teaching all of this to you in this
12:14
course so that the architecture aspect
12:17
of it the second aspect is the modeling
12:19
where we have to learn about
12:20
reinforcement learning DPO Po and grpo
12:25
group relative policy
12:27
optimization uh which is one of their
12:29
um major things which they implemented
12:33
to create um the reinforcement learning
12:36
pipeline we have to learn about the
12:38
modeling aspects step by step and only
12:41
after we learn about all of this then we
12:44
will truly know how every single element
12:46
of deeps is assembled from scratch and
12:49
towards the end of this playlist we'll
12:51
build something like this as I mentioned
12:52
we'll recreate deeps R1 and I'll show
12:55
you how to run it on gpus as
12:57
well as I mentioned I'm parall working
13:00
on a book where I'm I'm mentioning or
13:03
taking down all of these notes and I'll
13:05
also plan to make this book public along
13:07
with the videos but that will come one
13:09
or two months
13:10
later and simultaneously here are the
13:13
notes design notes which I've made I
13:15
plan to make this course as illustrative
13:17
as possible because from blog articles
13:20
it's very difficult to understand text
13:22
but I think if all of you understand
13:25
visually um I'll be able to explain
13:27
Concepts in a much better manner so I
13:29
spent a lot of time organizing this it
13:32
takes a huge amount of time and effort
13:34
on part of our entire team um but the
13:37
end goal is to help you build every
13:40
single thing here from scratch so when I
13:42
show you MLA I'll multi latent attention
13:45
I won't just explain the theory after
13:48
Theory we'll go into code and build the
13:50
multi-head latent attention module from
13:52
scratch and we'll compare it with the
13:54
traditional multi-head attention without
13:57
the latent part similar we'll do for
13:59
mixture of experts and all other things
14:01
here we'll understand the reinforcement
14:03
learning methods fully from scratch so
14:06
overall this course you can expect to be
14:08
around 20 to 25 hours or it can even be
14:10
more than that and I'll try to release
14:13
lectures as frequently as possible but
14:16
um it takes time for every single video
14:18
I have to curate notes like this and
14:20
then put it out there so that all of you
14:21
can
14:22
understand but the reason I made this
14:24
introductory video is to share the plan
14:27
of what this lecture series is going to
14:29
be all about what you are going to be
14:31
learning this is what you are going to
14:32
be learning so what all other people are
14:34
doing is that they are only focusing on
14:36
these three parts right which are
14:38
running deeps models with API calls
14:40
building rack based chatbots with deeps
14:43
or building llm applications with deeps
14:45
that's also fine we'll learn that but
14:47
that's probably 5% of the effort the 95%
14:51
of the effort will be to teach you
14:53
these the architecture and the modeling
14:55
because once you know the architecture
14:57
and modeling the application will seem
14:59
straightforward and easy to you that way
15:01
you'll truly become a strong llm or a
15:04
machine learning
15:06
engineer so again thank you so much
15:09
everyone and I look forward to starting
15:11
this journey of building deep seek from
15:13
scratch with you see you








