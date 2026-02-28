
0:05
hello everyone and welcome to this next lecture in the build deep seek from
0:10
scratch Series today my main agenda is to cover three things first we are going
0:17
to look at what exactly is deep seek second we are going to look at what
0:24
makes deep seek so special and why is everyone talking about it what are the
0:29
techniqu details which make deep seek so special and the third thing which we are
0:34
going to look at is what is the plan in this lecture series what we are going to
0:41
learn and what is the sequence in which we are going to learn about different
0:46
topics so let's get started uh first what is deep seek
0:52
essentially deep seek is a Chinese company which builds large language
0:58
models and uh before we dive any deeper first let me give you a quick overview
1:05
of what large language models essentially are all of you would have interacted
1:11
with chat GPD right I can go to um chat GPD right now and let's say I can ask
1:18
that make make a travel plan for me to visit
1:27
Italy and then chat G PT will come up with this travel plan this is a large
1:33
language model but one thing to keep in mind is that at the heart of it what a large
1:40
language model is is that it's an engine which takes in a sequence of words and
1:46
then it gives a probability of what's the most likely next
1:53
token and when such tokens are aggregated together it forms sentences
1:59
such as what we see here right now so essentially large language models
2:04
are probabilistic engines for predicting the next token this is the first thing to keep in mind here's a simple code
2:11
which I wrote to demonstrate that the models and their prediction are probabilistic in nature so for example
2:19
if the sentence is after years of hard work your effort will take you and we
2:24
have to complete this sentence I used the open API key and I um ran this code
2:32
which predicts the top 10 tokens and their probabilities so although the
2:38
output will be two which is essentially or which essentially implies that the next token
2:44
is two after years of hard work your effort will take you two but you will
2:49
see that there is also some amount of probability which is assigned to tokens like far places where
2:55
Etc so although you see these answers over here and they look deterministic it
3:01
seems that Chad GPT is very confident in giving these answers just keep in mind
3:07
that underneath for every single token there is a probability and we choose the next token with the highest probability
3:14
usually the distribution of the next token probabilities looks something like this on a log log scale it looks
3:20
something like this the first token generally has very high probability in the case of confident predictions and
3:27
then the and then the probability just just keeps falling down all right so that's the key idea
3:34
behind what large language models are and then what exactly is this large
3:39
behind large language models to put it simply no one really has an exact definition of what large is but
3:47
essentially there is a scaling law with size the first paper which kind of figured this out was the gpt3 paper let
3:54
me quickly show that paper to you so here's the gpt3 paper which was titled language models are few short Learners
4:02
where the authors essentially proved that if you increase the model size to something as high as 175 billion
4:10
parameters for both one shot and few short learning the model performance essentially dramatically improves I
4:17
believe that this was when we truly crossed the size barrier from 1.3 billion parameters to 13 billion
4:24
parameters to ultimately 175 billion parameters
4:30
and once this size barrier is crossed we start seeing wonderful properties of large language
4:35
models so this was gpt2 gpt2 the largest model had around 1.5 billion parameters
4:43
gpt3 the largest model had around 175 billion parameters and in general years
4:50
a paper which was published in nature right about neural networks and size so
4:56
from the 1950s to 2020 you can see that there is an exponential increase in the
5:03


***

number of parameters which we are using and recently you see the orange dots are more this is because language models
5:10
have really dominated the uh size space so essentially
5:15
language models and their size has been drastically increasing and we have reached about 1 trillion right now so
5:22
look at this straight line here which I've shown by this red arrow so the
5:27
y-axis is in log scale so this is an exponential scaling law which means that the size of the language models is
5:35
increasing exponentially why do we really care about size so much and why do we
5:41
increase the size well because people have seen that as you keep on increasing
5:47
the size of language models You observe something called emergent Behavior or emergent properties these are
5:54
essentially properties uh which are not present in smaller models
6:00
but they are present in larger models so for example if you see all of these tasks here such as performing
6:07
arithmetic um word unscrambling tasks Etc all of these tasks you see there is
6:12
a pickup point in all of these and on the x-axis you can think of this as computational power or roughly
6:19
equivalent to the model size so if the model size increases Beyond a certain point we have this pickup point so the
6:27
model suddenly starts learning about new things the model starts developing these magical awesome properties although the
6:34
model is trained simply on the next token prediction task as the size of the llm goes on increasing after a specific
6:42
size after a specific size the model can do these wonderful Properties or rather
6:48
the model shows these wonderful properties like could be translation summarization grammar checking Etc and
6:55
that's why we are in a race to build larger and larger and larger models companies like open AI anthropic have
7:02
even publicly said that they are just chasing size at this moment because there is still some hope that maybe
7:09
after 10 trillion or after 100 trillion number of parameters the llm shows some
7:15
properties which we are not aware of at all right now so these are the emerging properties
7:21
of large language models one thing to note is that llms differ from earlier
7:26
NLP models because earlier NLP models were essentially designed for specific
7:31
tasks such as language translation whereas due to such emerging properties
7:37
which we just saw large language models can do a wide range of
7:42
tasks such as translation summarization fact checking grammar checking Etc as
7:48
all of you might have explored with chat GPT right um just a key point to note
7:54
earlier language models could not even write an email from Custom instructions a task which is Trivial to Modern large
8:02
language models so until now we have seen that large language models really become
8:09
better and better with size they develop emerging properties and one more thing to note is that at the heart of this
8:16
language revolution rather is this architecture which is known as Transformers uh if you don't know what
8:23
Transformer architecture is don't worry we'll cover that in this series but essentially there was this paper called
8:29
attention is all you need which introduced a Transformer architecture it looks uh a bit complicated as shown in
8:37
this diagram to truly unpack the Transformer architecture it takes um a
8:44
couple of lectures but essentially this is the secret source which Powers language models and we are going to
8:50
learn about this so don't worry um finally one thing which I want to
8:55
mention is that when we say creating a large language model it inv involves two stages the first is a pre-training stage
9:03
where we don't have a label data set but essentially the model on its own creates
9:09
training data and the labels this is called as an auto regressive stage so
9:14
models which are pre-trained they are also called as foundational models so to do this pre-training we typically
9:21
assemble huge amounts of data from internet textbooks media research articles etc for example gpt2 was B on
9:30
data from Reddit books Wikipedia articles open web Corpus Etc and then
9:36
this whole this giant large language model is trained on this huge amount of data this training costs upward of
9:43
million dollars might even cost tens of hundreds of millions of dollars as the size of the llm
9:50
increases keep in mind that after pre-training the model develops basic capabilities and after that we typically
9:57
need to fine tune the model so at the second stage we fine team the model with label data set so for example you could
10:04
teach the model to translate by giving it some labels of how translation usually proceeds you can teach the model
10:11
to follow instructions by giving some instructions such as hey convert 45
10:17
kilom to meters and then the answer is 45,000 M that's a label data which you
10:22
give to the model so it learns to follow instructions GPT 3.5 which became the
10:29
product Chad GPD it was trained with reinforcement um or rhf which is
10:35
reinforcement learning human feedback so essentially there were human annotators which graded the output and that was
10:43
passed back as a feedback to the L this this is very important for us because
10:49
this finetuning is a stage where deep seek really changed the game completely
10:55








***


and we are going to come to that when we see what makes deep SE s speci but to understand what is to
11:03
follow after um or after this section it is very
11:09
important for all of us to be on the same page with respect to what are llms
11:14
what are emergent properties what's the secrets s switch poers llms and most
11:19
importantly the two stages involved in creating an llm and that is pre-training
11:25
plus F tuning with that let's go to the next section which is essentially what are
11:33
the llms built by Deep seek and uh how did it get so popular so let's take a
11:40
look at the different llms which were built by Deep seek so if you go to their website which is deep seek.com and if
11:47
you scroll down to the bottom in the research section you'll see the different versions of the llms which
11:54
they built they first started with a simple deeps llm then the significant
11:59
Milestones was building of this deep seek version two then they have deep seek version three and then after that
12:06
came deep seek R1 they released papers for each of these so for example here
12:11
you'll see the paper for deep seek origin 2 it's a 52 page paper here you
12:17
will see the report for deeps version 3 again 53 page paper and uh here you'll
12:23
see the paper for deeps car1 my goal in this series is that um there are several
12:31
amazing things which they have mentioned in these papers related to the architecture related to the
12:37
training um Etc I'm going to unpack all of that and I'm going to break it down
12:43
into modular lectures so that you don't have to read this report but rather if you just go through these lectures
12:50
you'll understand the nuts and bols of what is happening um the main model which caught
12:57
everyone's attention though is the Deep sick R1 because as we'll soon see um deeps R1
13:05
was a reasoning model which achieved comparable performance to open a STP
13:10
model that to at a fraction of the cost and plus it was open source these were
13:16
like two amazing things at the same time right deeps R1 which was released in
13:21
January of 2025 it was a reasoning model but it had
13:26
a comparable performance to open a a remember open AI is closed Source this
13:32
model was fully open source which means you could literally download the model and run it locally if you have a big
13:38
setup and uh secondly the API cost to this model is a tiny fraction of what
13:46
the latest open AI model costs and that's pretty amazing if you put these two things together open source plus low
13:53
cost that's pretty awesome right taking a look at the number of parameters as we saw the size scaling law language models
14:01
get better with size and deep seek was no exception deep seek version three has
14:06
671 billion parameters and then deeps R1 was a reasoning model which was
14:12
essentially constructed after deeps version 3 foundational model was built so version
14:18
two and version three are different foundational models which they have and R1 is the reasoning model which came
14:25
from Deep seek V3 so keep in mind this progression they first had the version one which they call DC KLM and then they
14:32
made DCM math DCM coder then they had deeps version two and then deeps version
14:39
2 coder then we had deeps version 3 and finally we had deeps R1 which really
14:45
broke the internet and because of which we are having this lecture series at this
14:50
moment all right so until now we have seen about llms we have seen about the
14:56
different llms which which were built by Deep seek now let's start getting into
15:01
the uh core content a bit which is first of all let's compare deep seek with
15:07
other AI models and let's see um is it better why is it better how low is its
15:13
pricing Etc so what I actually did was uh I'm very fascinated with this
15:19
mathematical question because it's an amazing problem here is an integral from 0 to 1 x 4 * 1 - x 4/ by 1 +
15:30
x² the cool thing about this problem which completely blew my mind when I solved this for the first time is that
15:37
the answer to this problem is 22x 7 minus Pi pretty awesome right first of
15:42
all many of us are already confused that oh I thought Pi is equal to 22 by 7 but
15:47
that's not the case 22 by 7 is actually greater than Pi by a tiny amount and
15:53
this integral captures that amount it's a beautiful mathematical puzzle which
15:58
was was by the way also um introduced in the 1968 Putnam
16:05
competition which is known to have very hard problems the proof was first deviced in U somewhere in
16:13
1944 and again this is a beautiful relationship right so what I did is I
16:18
went to deep seek and chat GPT and I asked them to solve this I went to GPT
16:24
40 I gave it this integral and I asked to choose the correct answer remember
16:31
option A is the correct answer right let's see what GPT did as you scroll down below it says that the correct
16:38
answer is 2x 105 which is option b so it spectacularly failed now I went to deeps
16:45
and I as the same thing solve this deep seek did this step by step and I got to the correct answer which is 22 by 7
16:52
minus Pi although this is just one example and by no means from one example
16:57
we can compare and contrast the two but this just shows that deep seat is pretty
17:03
awesome at difficult problems it got to the correct answer and by the way did it in like 10 to 15
17:11
seconds um all right so now let us do a bit of formal comparison of deep seek
17:17
with other AI models first of all let's compare deep seek with gp4 right in
17:24
terms of performance this is still debatable and it's hotly debated on several credit forums but generally the
17:31
consensus is that deep seek is similar or Superior to gp4 on several
17:38
tasks um but still performance- wise let's say you consider them to be at the same level but the huge difference is
17:44
with respect to cost right so let's say if I'm invoking a deep seek versus a GPT
17:51
model if you look at the pricing per a million tokens gp4 is around $30 and
17:57
deeps car1 is is like 055 it's literally a fraction of the cost in fact you can
18:04
even see the pricing here on the x-axis it's uh evaluation score or an
18:11
equivalent of evaluation score rather and on the y- axis I have pricing what
18:16
you see clearly is that if you compare GPT 40 mini and if you compare deeps version 3 deeps version 3 is has
18:25
much uh better performance and it's reasonably
18:31
priced um with respect to gp4 om mini but now if you compare deeps version 3
18:37
with GPT 40 you'll see that deeps version 3 has much higher performance on the y- axis it's higher and it has
18:43
significantly lower cost if you compare deeps version 3 and GPT 4 in fact deeps
18:49
version 3 seems to be vertically at the top right which means its performance is at is really very good and it's also at
18:57
the leftmost side compared to all of these other models which indicates that it's also relatively cheap and that's
19:04
what the community was extremely excited about here we have a highly performant model which is very cheap and you know
19:10
what's the awesome thing this is fully open source there is always a tension between open source and closed Source
19:17
models but here is an example of a model which has finally bridged the Gap we
19:22
have an open source model which is equally performant or even better than GPT 40 which
19:29
you can invoke or you can make an API call at a fraction of the cost in fact if you go to deeps website
19:37
and if you look at the different parameters you'll see that deeps version 3 outperform GPT 4 on almost all of
19:44
these parameters which have been considered over here so the main thing is that in terms
19:50
of pricing deep SE completely outweighs gp4 because it's literally a fraction of
19:55
the pricing which is there and most importantly deep is open sourced whereas gp4 is closed source which means you can
20:03
literally download deeps host it if you have a GPU or if your machine is big uh
20:09
you can host it and run it but remember it's like 6 671 billion parameters so it's not quite easy to do
20:16
this secondly if you compare deeps with llama for example remember llama is a
20:22
host of Open Source model which have been released by meta amazing models but they are not not
20:28
quite as performant as deep seek deep seek has great scale and performance its
20:34
foundational model has hundreds of billions of parameters and it has strong results which really exceed what Lama 70
20:42
billion parameter can do so if you look at this plot again there is also Lama
20:47
here right and if you see in terms of performance deeps really outweighs if you see on the y axis deeps outweighs
20:53
all of the Llama models it outweighs the 70 billion instruct the 405 billion instruct because it has 6 671 billion
21:01
parameters not just that it has a lot of other Innovations in the architecture which we are going to see in a moment
21:08
mixture of experts reinforcement learning training multi-head latent attention multi- toal prediction
21:15
quantization in the input Etc all of this really give deep seek and Edge or
21:21
llama um all right so this is with respect to comparison of deep seek with
21:28
uh GP 4 and with respect to llama so in terms of strength and weaknesses deep
21:34
sick is pretty awesome with respect to cost efficiency its performance is also quite good and it's open source these
21:41
are the three biggest strength but the biggest weakness is that it might not be as polished or as safe as
21:47
gp4 in my opinion with further versions this might get tweaked so deep seek
21:52
might actually become safer or more polished but right now that is a bit of a concern for big corporations to really
21:58
implement this secondly it's 6771 billion parameters right so let's
22:05
say if you are an organization who does not want to use open AI because you have to make an API call so you're not
22:10
comfortable with your data going somewhere else so you decide to download and Host this model
22:16
locally that might take computational resources because it's a pretty large model 671 B billion parameter is not
22:23
tiny not easy to deal with so you need to figure out the Computing infrastructure for this but it will give
22:30
you data privacy if you host it on your own server let's say as an organization you must weigh
22:36
these factors right if you really care about safety guard rails maybe stick
22:41
with gp4 for now and soon change to DSE if you are a lean fast growing startup
22:47
go with deeps because it will tremendously cut down costs um it is highly performant and of course it's
22:53
open source if you want privacy which means if you don't want your data to be going to close Source companies again
23:00
then that would give the open source nature of deep seek and Advantage for you now let's come to the main point
23:08
which is the next section of this lecture what makes so special or what is so special about deeps how is it able to
23:15
literally charge people so less um how does it achieve so much cost efficiency
23:23
and still be competitive in performance with GPT 4 so I believe there are four
23:28
major things which we need to talk about here the first is that deep seek has an Innovative architecture second is that
23:35
the training methodology is very creative and Innovative third is that they have implemented several GPU
23:42
optimization tricks and fourth is that they have a model ecosystem which favors distillation Etc we'll see about all
23:49
four of these first of all in the architecture itself I believe that deep seek has done the following five things
23:56
right which make it truly an Innovative architecture first they have something called multi-head latent attention they
24:03
have mixture of experts model then they have multi toen prediction then they have
24:09
quantization and finally they have rotary positional encodings we are going to learn about
24:15
all of these in detail in fact going into each one of them itself we'll take two lectures there is very limited
24:21
information about all of this on the internet and it just you need to spend a lot of time on each of these to truly
24:27
understand the architecture Innovation here let me give you a quick flavor of what each of these actually
24:34
mean right so if you go to the original attention mechanism which I showed you in this paper which is attention is all
24:40
you need there was the multi-head attention mechanism which looks something like
24:46
this uh which looked something like what you're seeing in the figure right now
24:52
right this is the multi-ad attention mechanism which was there now deep seek what they did was they introduced
24:58
something completely different in order to make sure that the attention mechanism is implemented effectively
25:05
they had a key value cache but instead of a normal key value cache they had a key value cach in a latent
25:12
space um don't worry if you don't know what these terms mean right now we will we will cover all of this in subsequent
25:19
lectures but just make sure that uh or for now you should understand that they did this special type of caching special
25:28
type of key value caching in the latent space so that uh the attention mechanism
25:34
computation becomes much more efficient it takes up less space and it's computationally also fast that's one
25:40
change which they made so this was the multi-ad latent attention which was the first thing which we discussed the
25:47
second thing was with respect to mixture of experts so if you see the normal
25:53
attention mechanism it looks something like this right there is a feed forward
25:59
there is the attention layer and followed by feed forward neural network after the attention layer and the feed
26:05
forward neural network looks something like this in a mixture of experts model what you essentially do is that you have
26:12
four experts the entire neural network is not activated at once only parts of
26:17
the neural network are activated so there is a special routing mechanism which which actually decides which part
26:25
gets activated and which part does not get activated this is the router which essentially
26:32
decides which expert is going to get activated which expert is not going to get activated we are going to learn
26:37
about all of this also in a lot of detail this is another key Innovation which they implemented third is multi
26:44
token prediction as we saw at the beginning of this class that usually we
26:49
just predict a single token right they implemented uh a new thing which was
26:55
discovered in a paper released just last month that why don't you you predict multiple tokens instead of one token
27:01
what if that speeds up the process makes it more efficient then fourth thing is the implemented quantization so instead
27:08
of representing every parameter as a large floating Point number you just represent it uh in a slightly compressed
27:15
manner the best way to think about quantization is like here right in the original image on the left hand side
27:21
there are huge number of pixels whereas on the right hand side the image is just constructed out of eight colors
27:29
so it's kind of pixelized so if you zoom in it's completely pixelized see here it's pixelized over here compared to the
27:35
left hand side which is uh very sharp but if you zoom out you'll see that it
27:41
almost looks the same right so this is called as quantization and they implemented quantization in the
27:46
parameters of the Transformer block and then finally they have something called rotary positional
27:53
encodings so uh let's let me show you the three positional encodings right now
28:00
essentially in the attention mechanism which was published in 2017 They just added positional
28:06
encodings to token encodings which polluted the embedding Vector but soon
28:12
after that people realize that why don't we just rotate the original Vector the
28:18
query and the key vectors to capture the effect of positional encodings so that will not change the magnitude and that's
28:25
a highly efficient way to encode positions so instead of encoding positions in the
28:31
token embedding itself we encode it a bit later in the query and in the
28:37
keys uh so yeah this is with respect to the rotary positional encoding which
28:45
which is the fifth key Innovation which the deeps architecture implemented we
28:50
are going to look at all of these in a lot of detail now this was the first aspect which was Innovative architecture
28:56
but they did not stop there I believe the Deep seek paper truly makes the field of reinforcement
29:03
learning reborn because instead of just relying on human label data like we saw
29:09
for GPT 3.5 where humans graded the quality of the data and that was fed
29:14
back to the training process what they did is they used large scale reinforcement learning to teach complex
29:21
reasoning to the model and instead of human label data they had a rule based reward system so it was not relying on
29:27
human judgment but purely it was a rule based system through this they implemented a framework which is called
29:34
group relative policy optimization which is at the heart of the reinforcement learning training mechanism which they
29:40
implemented we are going to learn about all of this which this is the main reason why deeps car1 is such a good
29:47
reasoning model so reinforcement learning is a major part of what we will be covering
29:54
the third aspect is GPU optimization tricks now this is a a bit tough to understand so I'll not be spending too
30:00
much time here even in the course but essentially what they did was instead of using Cuda they used something called
30:07
Nvidia parallel thread execution the simplest way to think about this is that
30:13
uh in fact I asked at GPT what's the simplest way to think about this so if you think of Cuda as writing python or
30:19
Java code PTX is like one lower below so it's an intermediate step before machine
30:25
code execution and that just speeds up things a lot more so in high level
30:30
programming you are not at the machine code level right if you at the machine code level it's the fastest like C or
30:36
C++ whereas python or Java takes you to a high level that slows down things you can think of PTX as somewhere in the
30:43
middle whereas QA is at the higher level but PTX is at the middle a bit closer to the machine code execution so that
30:50
speeds up things a lot more there was also a nice article published about this yeah deep six AI
30:58
breakthrough bypasses industry standard Cuda for some functions using nvidia's assembly line assembly like PTX
31:05
programming instead so I believe that also played a major role in
31:10
U um in speeding up um their or making their architecture
31:16
a lot more efficient and which ultimately reduced the costs and finally they have a strong model ecosystem where
31:23
although the main model is 6 671 billion they have distilled it down to smaller
31:28
models even as low as 1.5 billion that makes it a pretty awesome model
31:33
ecosystem we are going to look at model distillation also so just to recap these four aspects which make deep seek so
31:40
special first is their Innovative architecture second is the training methodology which is centered around
31:46
reinforcement learning third is the bag of GPU optimization tricks which they have and fourth is model ecosystem
31:54
especially distilling a from a larger model into smaller models as small as
31:59
about 1.5 billion parameters now let's come to the last
32:06
two sections of today's class first is why is deep SE such a turning point in
32:11
AI history and I believe it is a turning point because of the following reasons it is the first time that a small
32:18
Scrappy startup has reached parity with the best AI models using novel
32:24
techniques and far less resources they have slashed down the development cost although it's not as low as $6 million I
32:31
believe it's still quite low compared to what big corporations like GPT or metas
32:36
Lama Etc the cost requirements which they use to train the models so it's
32:43
like it it was the first proof that even small startups even companies which are
32:49
not maybe as funded as open a can build a large language model which is which is
32:55
awesome right which performs at parage with gp4 it and it does so with far less
33:04
resources what also happened with this is that people got scared right investors got
33:09
scared um there was a huge dip in the US tech US tech stocks in January 2025
33:15
because of deep six advancement the main reason was the idea of a lowcost open source Chinese AI
33:23
model threaten The Profit models of open AI Microsoft and even Google
33:28
and raised concerns about the AI supply chain and GPO markets one model or one company deep
33:36
seek brought about so many changes and that's what makes deep seek a turning point in history one major thing which I
33:43
believe has also happened because of deep seek is that developing countries such as for example India have started
33:49
have heavily started investing into building their own large scale foundational models if China's deep SE
33:55
can do it then why not other countries why only us and why only companies
34:01
coming out of us rather right why not other companies with resources which deep SE used can build their own
34:07
foundational models so all of this discussion has been started in fact the Indian government has also U released a call
34:15
for building foundational models that's pretty awesome I think and it's one of the main motivating factors for me to
34:21
create this series now let's come to the last section of today's lecture which is our
34:27
plan for this lecture series we have developed or I have rather divided this
34:32
lecture series into four phases based on what is the speciality about deep seek
34:38
the first phase for us is going to be um going into the architecture so
34:44
first I'll start with the attention mechanism then I'll go into multi-ad latent attention mixture of experts
34:50
multi token prediction quantization and rotary positional encodings I am going to assume that you have some amount of
34:56
knowledge of attention if not you can check the build llm from scratch Series so this series is going to be a bit more
35:03
advanced and it assumes that you you go through that previous series before I am going to start at a slightly higher
35:09
level here then we are going to go in the phase two in the training methodology phase three GPU optimization
35:16
tcks this will be a small phase I won't be having too many lectures here and then I'll conclude with lectures on
35:23
distillation a bulk of the lectures will be on phase number number one and phase number two and smaller number of
35:30
lectures on phase three and phase four so this is the main plan which we'll be following for the
35:36
series let me quickly summarize what we learned today first we looked at large
35:41
language models and the fact that they are engines of probabilistic next token
35:47
prediction we saw that size is a very important factor in large language models there is a size scaling law as
35:54
the size increases the models get better and better they start developing emerging emergent properties which are
36:00
not present in smaller models then we saw that the llm secret source is essentially Transformer the Transformer
36:08
architecture and then finally we saw that creating an llm means building a foundational model which is essentially
36:15
the pre-training stage and then we have a fine tuning stage there are two parts
36:21
then we saw that although deeps R1 has become popular the deeps company started
36:27
long back they started with the deeps llm first which is version one then they had version two and ultimately version three
36:34
which was a huge model 671 billion parameter and then ultimately they made deeps car1 which is a reasoning model
36:41
and that broke the internet why did it break the internet because deeps car1 U
36:47
has comparable performance to open AI stop model and at a fraction of the Cost Plus its open
36:54
source so deep seek is equally performed as GPT 4 their pricing is literally I
37:00
think 100 to 500 times less as I showed you in this lecture and finally it's fully open
37:07
source um strength and weaknesses the biggest strengths of deep seek are that
37:12
it's open source it's cost efficient and it has competitive performance so three big strength the biggest weakness might
37:19
be that it's not maybe as polished or safe as let's say gp4 or BL another weakness is that if you're
37:26
planning to deploy it locally or planning to use it securely you need to have infrastructure for downloading and
37:32
using a 671 billion 671 billion parameter model then we saw what makes
37:38
deep seek so special and there are four key ingredients here the first is the Innovative architecture training
37:45
methodology GPU optimization and model ecosystem within the training or within
37:51
the Innovative architecture we have five key things multi-head latent attention mixture of experts
37:58
multi- toen prediction quantization and rotary positional encodings then uh in the training
38:05
methodology we have the fact that they used large scale reinforcement learning to teach complex reasoning to the model
38:12
and they used a rule-based reward system uh which is also known as group relative
38:18
policy optimization rather than relying on human label data in the GPU optimization tricks they
38:25
used parallel thread exec PTA instead of Cuda only in some places I believe and
38:31
then finally they have a strong model ecosystem where they distill their main model into smaller models as low as 1.5
38:39
billion parameters in this lecture series we are going to follow the same workflow we'll
38:45
go with the first phase which is innovative architecture then we'll go to the second phase which is training
38:51
methodology then we'll go to the third phase which is GPU optimization tricks then we'll go to the fourth phase which
38:58
is model ecosystem I am going to assume a a good amount of knowledge about llms
39:04
and I will explain the attention mechanism again but I'll essentially start from the attention mechanism and
39:09
then dive into the details if you're a complete beginer I recommend the build llm from scratch series
39:17
first um and then finally I believe deep seek is a turning point in history because they literally showed that even
39:24
developing countries can build their own foundational model um if we are smart about the
39:31
Innovative architecture if we are creative um we can build a foundational model which is as good as let's say open
39:39
a models and that to at a low cost
39:44
and fully open source so they are truly democratizing AI that way so thanks a
39:50
lot everyone and uh I look forward to seeing you during the next lecture thank
39:55
you
