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


started to get pretty crazy after that
5:07
so here was a LinkedIn post which was
5:09
released which said that deep seek had
5:12
erased $2 trillion worth of market cap
5:15
in US
5:16
Stocks companies in the US lost a huge
5:19
amount of money and the reason was
5:22
because China's deep seik built a
5:25
reasoning AI model that Rivals open ai's
5:28
best model and that to it did it for
5:31
1,000 of the cost and more than that it
5:34
was open
5:35
source deep seek was not just cheaper to
5:38
build it was also cheaper to
5:40
run so this was more than technical it
5:44
was also political thing right so had
5:46
silicon valy lost its AI Advantage deep
5:49
seek raised all of these questions it
5:52
was around 27 times cheaper than open AI
5:55
it was open source and permissively
5:57
licensed it's just amazing
6:01
it led many countries to act also after
6:03
that so for example India released a
6:06
proposal which was a call for building
6:09
the nation's first foundational model so
6:12
people were suddenly like if China as a
6:14
country can do it with few resources
6:17
compared to the US why not other
6:19
countries also so they started this
6:21
debate
6:22
also and then I got really curious about
6:25
this and I started searching a lot about
6:28
deep seek I I always like to build
6:30
things from scratch there is this build
6:32
llm from scratch playlist which I
6:34
released which has now around 4045
6:38
videos and it has received a huge amount
6:40
of interest for people and I saw this
6:42
one post where someone had tried to
6:46
replicate um deep SEC car1 it was just a
6:49
small tutorial but this ignited the idea
6:52
in my mind that why don't I try to build
6:55
deep seek from scratch or why don't I at
6:58
least understand how every single module
7:00
of deep seek was built from scratch and
7:03
then I started doing a bit of research
7:04
right and I realized that deeps R1 was
7:07
not the first thing which came out of
7:09
this company in January 2024 they had
7:12
this deep seek llm paper then in Jan
7:15
2024 again they had deeps coder then in
7:19
March of 2024 they had a vision language
7:23
model in April of 2024 they had deep
7:27
seek math which was related to
7:28
mathematical reasoning in open language
7:30
models then came deep seek version two
7:33
in June
7:35
2024 then deep seek coder version two in
7:38
June
7:39
2024 then finally came along deep SE
7:42
deep seek version 3 and this really blew
7:45
everyone's mind because this was the
7:47
foundational model which ultimately led
7:49
to this paper called Deep seek R1 which
7:52
only came out in January of 2025 so
7:55
there is one year of research or maybe
7:57
more than that to reach to this stage
7:59
where it finally came into the public
8:02
Limelight and all of these Innovations
8:04
which happened in these papers right all
8:06
of these papers I really wanted to dig
8:08
deeper and I wanted to understand I knew
8:11
exactly how llms were built from scratch
8:13
but deep seek was a revolution on top of
8:16
the foundational llm models and the
8:18
architecture which we have traditionally
8:19
seen so I wanted to learn everything
8:22
about it and naturally I started
8:24
searching on YouTube first related to
8:26
deep seek from scratch then I could see
8:29
here's a 20 minute video which is around
8:31
1.5 million views here is a video which
8:35
is again 21
8:37
minutes and it has huge number of views
8:39
but you can see that all of these videos
8:41
are either 10 minutes or 15 minutes this
8:44
is another video which is just 8 minutes
8:46
this is not what I was looking for at
8:48
all I wanted to know the nuts and bolts
8:51
of every single piece of how deep seek
8:54
was built and assembled from scratch
8:56
it's as if I want to build a sports car
8:59
all by myself and I'm just being shown a
9:01
5 minute version of how a sports car
9:03
looks like that's not what I was looking
9:05
for I wanted a video which explains the
9:09
mathematical details which explains the
9:11
code from scratch and which takes me
9:14
through the building blocks right
9:15
similar to the way I did for build llm
9:17
from scratch but I just could not find
9:20
anything on YouTube then I started
9:22
searching on Google I could see that
9:25
several people are building apps using
9:27
deep seek that is fine uh building apps
9:30
with something already existing is cool
9:33
but I really don't want to do that I
9:35
want to be able to build deep seek on my
9:39
own that's where the real power lies in
9:41
my opinion not in building applications
9:43
on top of something which already
9:46
exists then I searched on internet again
9:48
build deep seek from scratch and I could
9:51
see several forums several articles like
9:53
The Ultimate Guide to DEC car1 but here
9:56
again it it already uses DEC car1 and
9:59
apps on top of that so this was not what
10:03
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







