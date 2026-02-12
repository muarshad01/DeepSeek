* [Invideo AI - Video Generator | No Editing Skills Needed](https://invideo.io/?utm_source=google&utm_medium=cpc&utm_campaign=Top16_Search_Brand_Exact_EN&adset_name=InVideo&keyword=invideo&network=g&device=c&utm_term=invideo&utm_content=InVideo&matchtype=e&placement=g&campaign_id=18035330768&adset_id=140632017072&ad_id=616240030555&gad_source=1&gad_campaignid=18035330768&gbraid=0AAAAACqfi_CasH_dati6efWraWC4sWV3x&gclid=Cj0KCQiA7rDMBhCjARIsAGDBuECUPsCNxYndLvG9dbQcl3duZVnCdxZL4bu-YboI7x4VTCTYt6KBjfcaAvMFEALw_wcB)



***

#### Part-1: Innovative Architecture
* Multi-head Latent Attention (MLA)
* Mixture of Experts (MoE)
* Multi-token Prediction (MTP)
* Quantization
* Rotary Positional Encoding (RoPE)

***

#### Multi-head Latent Attention (MLA)
We need to understand the following concepts to truly understand MLA:
* Architecture of LLM
* Self-Attention
* Multi-head Attention
* Key Value (KV) Cache

***


process first we are going to understand the architecture of llms itself and
6:08
that's going to be the main purpose of today's lecture I believe that without having an intuition of the llm
6:15
architecture it's impossible to understand latent attention then we are going to
6:21
understand why there was a need for self attention and what is the self attention
6:27
mechanism itself once we understand self attention we are going to understand how self attention
6:35
was transformed into multi-head attention and what does it mean to have multiple attention heads that's the
6:42
third aspect over here and then the fourth aspect is essentially key value
6:47
cache so then we are going to understand okay multi-head attention works and it
6:53
really works very well but then what can we start doing to improve the efficiency
7:00
of multi-head attention to make it computationally faster to make sure that the number of parameters which you are
7:08
storing in the memory is reduced and that's when key value cache comes into the picture only when you truly
7:16
understand key value cache then you will slowly start to understand multi-head latent attention so after KV cache then
7:23
we are going to understand MLA but I'm going to devote a lot of
7:29
time to develop your foundations in these first four Concepts before we go
7:35
ahead to MLA itself many of these blog posts which I mentioned right now they assume that you
7:41
already familiar with this so in the build deep seek from scratch series we had 43 lectures
7:48
explaining you everything about these different aspects the in this lecture series I'm not going to go as deep so
7:54
for example today I'm going to have just one lecture on the llm architecture but in the build llm from scratch there were
8:01
three to four lectures right so my aim is that I want to
8:07
explain this knowledge to you but also I want to make sure that beginners who have started watching this series they
8:14
also feel connected with the series so the challenge for me was to come up with a new set of lecture notes specifically
8:22
for this series because I'm trying to explain a concept in a in 1 hour but also so I want to
8:30
make sure that I don't lose out beginners in the process so I've made a whole series of
8:36
new notes for explaining these different concepts all right so I hope everyone
8:43
has understood the flow of how we are going to understand multihead latent attention today our main aim is to
8:51
understand the architecture of llms so after today's lecture all of you
8:56
should have a mental map or a visual Road map of what happens with a word or
9:02
a token when it goes into an llm first of all let me explain what
9:07
does it mean architecture of an llm right so if you pass in some certain a
9:13
sentence or a sequence of words we have seen that when a sequence
9:18
of words is passed into an llm what the llm essentially does is that it predicts
9:23
the next word or it predicts the next token rather so an llm or a large Lang
9:29
anguage model can be thought of as the next token prediction engine right so it can be thought of as the
9:37
next token prediction
9:43
engine and just like an engine so let's say if I go and search engine car and uh
9:51
if I copy this right now copy image or let me copy a good image right now if I
9:57
copy this image right now and if I paste it over here right this is an
10:04
engine so if we are calling it the next token prediction engine we need to know
10:10
how does this engine actually work how does this engine actually
10:18
work in the previous lecture we have learned some things about the engine
10:23
what have we learned the first thing which we have learned about this engine is that it has a huge number of parameters so to give you an example
10:31
gpt3 has around 175 billion parameters whereas uh GPT 4 although it's not been
10:39
released yet they probably have around a trillion parameters GPT 4.5 which was
10:45
just released 2 to three days back maybe have 5 trillion or 10 trillion parameters that's not yet released but
10:51
we know that this engine has a huge number of parameters which are acting together and if you search engine car
10:59
working right you'll see that there is this piston cylinder mechanism here
11:05
right and uh the Piston cylinder mechanism essentially works and then that's how the engine
11:11
operates similarly we need to make sure that we understand how do these
11:17
parameters actually work where are these parameters can we open the engine of the
11:22
large language model and try to see what happens inside that engine essentially
11:29
given a sequence of words so similarly to a car right when a fuel is injected
11:35
into this engine what essentially happens with that fuel and how is that converted into the motion of the car
11:42
similarly when a sequence of words is passed to this engine what's
11:47
underneath the llm which causes us to predict the next word so think of the sequence of words
11:55
as a fuel and think of the llm of course as the engine and the next word as the
12:00
motion of the car let's say so here we want to open the engine right and to
12:06
truly understand how the engine actually works we need to understand the architecture of the
12:11
engine which means that we need to understand what are the different components inside that engine how these
12:19
components are actually connected to each other then what exactly happens
12:25
with the fuel or the sequence of words when it is being passed through this
12:32
architecture that's what we essentially want to understand in today's lecture so you can think of today's lecture as
12:39
going to your car opening the engine and trying to Peak inside the engine and
12:44
trying to really understand how do you work uh one way to also think about this
12:52
is that all of us have worked with chat GPT now right and if we ask something like
12:59
uh give me an essay give me a short essay on
13:06
friendship right and we see that GPT essentially predicts one token at a
13:12
time how does this actually work what's the architecture which is giving me this
13:17
one token at a time prediction that's what we are going to find out today okay
13:23
I'm going to try to keep this lecture as beginner friendly as possible so you'll see that there's a whole story which
13:28
I've constructed to explain how the llm architecture works I don't think this
13:33
has ever been done before so it's also a bit of an experiment for me to explain this to you in a story kind of a format
13:41
but I hope through this explanation you'll understand really how the llm architecture is
13:47
working okay so here's the schematic of what happens when you take a look at
13:54
the engine what happens when you open the Black Box what lies within the black Black Box itself and when you open the
14:01
black box you'll see that there are a huge number of things which pop out it's
14:07
not simple at all uh if you think of the 175 billion parameters the 175 billion
14:13
parameters are scattered across multiple places of this black box in other words
14:19
the llm architecture is quite complex um and why do you think it might
14:25
be complex well because in making the next token prediction the llm is
14:31
actually learning language itself right I strongly think that language learning
14:37
is a byproduct of the next token prediction task and to learn the language you cannot have an engine which
14:44
is small or you cannot have an engine which is not complex enough so that's
14:49
why our engine is quite complex we have a huge number of blocks or layers which
14:54
are linked together today we are going to try to understand this architecture so if you just take a look at this
15:01
schematic you'll see that broadly this schematic is divided into three parts there is this the there is this part
15:08
number one over here part number one then there is the part number two which I marked by saying
15:15
that it's something which is called as a Transformer block which let me Mark like this and
15:22
then there is a part number three which is basically the part which is the output part so yours
15:28
where the next token is actually predicted so the architecture of the llm
15:34
can be thought of in three parts the first part can be thought of as the input the second part can be thought of
15:41
as the processor and the third part essentially can be thought of as the output right in
15:48
the input part we have the sentence let's say if you have the sentence any sentence let's say the next day is
15:55
bright there are a number of things which happen with that sentence and with every token or every
16:02
word in that sentence before we pass it to the
16:07
processor and the things which happen to the sentence are first of all we do something which is called as
16:14
tokenization um secondly we do token embedding and third we do something called as positional embedding after
16:21
these three steps are done the input is essentially passed to the processor which is also called as Transformer
16:28
block within the Transformer block there are six different things the normalization layer multi-head attention
16:35
layer Dropout second normalization layer feed forward neural network another
16:40
layer of Dropout these two plus signs here are what are called as skip connections or shortcut
16:48
connections finally when we come out of the processor or the Transformer block we have the output and here we have
16:55
another layer normalization layer and the final layer for the next toen prediction okay so all of what I said
17:02
right now if you are learning this for the first time you might be thinking what is going on here and all of this
17:08
seems too complex let's break it down further and that's exactly what I'm
17:14
going to do right now but keep or keep close eye or attention on this part
17:20
which I marked as purple because the first Innovation which I'm going to explain later which is the multi-head
17:26
latent attention or MLA here is to do with this aspect which is
17:32
titled as multi-ad attention so out of the entire architecture there are two major places where deep seek has
17:39
contributed in making their Innovations the first is this called as multi-head attention that block and the second is
17:46
the feed forward neural network so let me actually rub this a bit right
17:54
now so the MLA or the multi-ad attention is an in inovation which happened in
18:00
this part of the architecture so I'm going to call this MLA and the mixture
18:05
of experts Innovation that actually happened in this part of the architecture Moe so again if you don't
18:12
understand the architecture itself you will not be able to appreciate where the innovation has happened but it's like
18:18
imagine opening an engine right and you have opened the Deep seek engine right now and the Deep seek engine is that of
18:25
a car which has performed very well and you want to understand why has it performed very well you open the engine
18:30
you see all these parts and now I'm telling you there are two parts in this engine which were augmented to improve
18:36
the performance but all right coming back to the engine itself how do the input processor and
18:44
the output actually work and now my challenge for this lecture was that I wanted to create one lecture in which I
18:51
explain all of this to you I explain the input I explain the processor and I explain the output and I wanted to
18:57
explain it in a e easy to understand manner so that you get an intuitive feel for the architecture itself so the way I
19:05
thought of doing this was that I thought from the perspective of
19:11
the fuel so if you are a fuel let's say and if you go to the car engine what
19:17
happens with you do you go to the engine first then are you rotated because of the Pistons and then some kind of a
19:24
power or energy is produced if I understand the life of the fuel I'll
19:30
effectively understand how the engine works right similarly today I want to
19:35
show you the life cycle of a single word and what happens to a single word
19:43
when it essentially goes through the llm architecture think about it this way right when we when we put this sentence
19:51
give me a short essay on friendship or let's say if we are going to complete the next sentence and the sentence which
19:58
I'm going to take is let's say the next day is bright the
20:05
next day is bright it goes into the llm
20:14
engine and then the next token is predicted let's say the next token is and the next day is bright and what I
20:21
want to show you is we are going to focus on just one token and we are going to see what happens to this token as it
20:28
goes through every single step of this llm architecture and by looking at from
20:34
the perspective of the token so I want you to now imagine that you are that token you are that token and I will now
20:41
take you through what happens to the Token as it goes through several of
20:47
these layers in the llm architecture itself and ultimately we predict the next
20:54
token so that's how I'm going to explain this whole lecture to you in a story format I'm going to explain this lecture
21:01
to you as if you are the token now imagine you are the token you're surrounded by a bunch of words and
21:07
suddenly you are thrown to the llm architecture let's understand the life cycle of a single token so that's how
21:13
the next part of this lecture is going to be all right so let's embark on this
21:20
journey together in which we will understand how the life of a token
21:25
essentially looks like so the the title which I given to this section is the
21:30
Journey of a token through the llm architecture so what I first did is I
21:36
went to chat GPT and I asked it to write a short paragraph on friends right so I took some random sentence from this
21:42
which is a true friend accepts you let's say we are looking at this
21:48
sentence um a true friend accepts you which is a sequence of five words let's say that's my input sequence at the
21:55
moment and we have to predict the next token given this input
22:00
sequence and I'm going to specifically focus on the word friend and I'm going
22:06
to think from the perspective of friend now put yourself in the shoes of this
22:11
word or this token uh first of all this token so I'm
22:17
going to interchangeably use token and word although they are not the same for
22:22
the sake of Simplicity I'm just going to say one token is equal to one word so put yourself in the shoes of this token
22:29
now what do you see you see that there are these other tokens around me right there is a true accepts you there are
22:36
these four other tokens which I'm used to hang out with just like friends hang out together I'm used to having these as
22:44
my neighbors and my friends a true except send you these are the four
22:49
neighbors of this token which we have chosen that is friend now suddenly what happens in the
22:56
first step is that the first step in the llm architecture is that that's the step
23:01
of isolation So currently we are looking at this phase right which is the input
23:07
phase right um the first step which happens in the input phase is the isolation phase
23:14
so what happens is that the word is detached from its neighbors the word is
23:19
isolated from its neighbors so imagine like a group of friends and every person is isolated from their neighbors so this
23:27
word is isolated and we look at it in isolation that's phase number one phase
23:33
number two is essentially called as token ID assignment which means that
23:38
imagine now every word is isolated and we want to put a badge or a stamp on
23:44
every word or every token similar to how let's say if you are getting enrolled in a camp or
23:50
enrolled in military or any other group activity you are given ID that's your role number or in school all of us have
23:58
role number numbers right it's similar to that so every token is isolated and then it's assigned a separate token ID
24:05
the way the token ID is assigned it's I'm calling it getting your
24:10
badge the way the token ID is assigned is a very interesting process we have a
24:16
book of token ID so you can think of this like a encyclopedia or a book of
24:22
token IDs in this book basically all the possible tokens are listed all the
24:27
possible to tokens are all the possible words and then there is a number which is associated with every
24:34
word u in this book there are not just words there can be characters or there can be even subwords so this book
24:41
consists of characters like a b up to zed it even consists of subwords such as
24:47
maybe C can be a uh word in this vocabulary then it may consist of
24:53
subwords like uh isation that can be a sub word
24:59
and then it also consists of words like let's say token can be um a full word in
25:05
this vocabulary enter can be a full word in this vocabulary begin can be a full
25:10
word in this vocabulary so you can think of this book of token IDs as consisting
25:16
of characters words and subwords so let me write this down here that's going to
25:21
be very important this book of token ID is essentially consists of
25:28
it consists of characters it consists of words and it also consists of
25:35
subwords so as a result essentially we make sure that every token or every word
25:42
which is isolated it finds certain badge there is no token or no word which is
25:48
isolated which won't find any badge readers who are familiar with the
25:53
concept of bite pair encoding remember that to create this book book of token
25:59
ID itself there is a certain scheme which is called as a bite pair encoding
26:04
scheme this is a subword tokenization scheme and to create this book of token
26:11
ID we use this scheme so gpt2 for example relied on the bite pair encoding
26:16
mechanism to create its
26:24
vocabulary this vocabulary is this book of token IDs is also called
26:30
as the vocabulary and then it changes from one large language model to another
26:35
so let's say gpt2 has a vocabulary of 50,000 GPT 4 might have a vocabulary
26:40
Which is higher maybe 100,000 right so based on the LM which we are using the
26:45
token ID which is assigned to let's say this friend might change so I am using a
26:51
large language model here right now which has a vocabulary size of 50,000 which means that there are 50,000 tokens
26:58
which might be a combination of characters words and subwords all right then what I'm going to do is that
27:06
I'm am going to essentially look at this vocabulary and I'm going to find where the friend comes into the picture
27:14
and I'm going to find the token ID associated with it right so for the word friend the token ID which is associated
27:21
now is 20112 so I'm going to note that down uh
27:27
so that's the badge or that's the role number which is now assigned to this token so the role number assigned to the
27:34
Token friend is now 20112 similarly all the other tokens or
27:39
all the other wordss will get a similar badge that's the first step or rather that's phase number two which is token
27:45
ID assignment so now imagine that this token friend which was isolated from its
27:51
neighbors it has now been given a badge or a stamp which is 20112 that's phase number two I did not
27:59
go into the details of how this book of token ID was created because if you want
28:04
more details on this there's a separate lecture on creating this vocabulary itself for every large language model
28:11
and it's called bite pair encoding from scratch it's present in the lecture series build llm from
28:17
scratch but for now stay with me imagine you are this token friend you have been
28:23
isolated and now you have been given a badge or you have been given a role number
28:28
then you essentially come to phase number three in Phase number three something interesting essentially
28:35
happens in Phase number three the you until now you just had one number
28:40
associated with you right but now you are going to have a huge Vector of numbers which are going to be associated
28:46
with you and this is called as token embedding assignment one way to think of this is
28:52
that let's say we have an entrance examination which has 7
28:58
168 questions and each question essentially tests a certain feature of you so now we are looking at the word
29:05
friend right each question will test are you a noun are you a gender are you a verb are you a sport are you an emotion
29:13
Etc we actually don't know what these features or what these questions are but I'm just trying to explain you so that
29:20
explain to you so that you get an intuition of what token embedding is so imagine there are 768 questions like
29:27
this which are asked to every token which we have isolated and then based on the answers
29:33
we get to understand something about that token whether it's a noun whether it's a sport whether it's a adjective
29:42
whether it's something which appears always at the end of a sentence whether it's something related to gender whether
29:48
it's something related to monarchy like kings princess Queens
29:54
Etc so here we are actually getting to know about the meaning of the token
29:59
itself in the token ID assignment we did not get to know anything about the meaning but in phase three in token
30:07
embedding because we ask a big list of questions we get to know something about the meaning and based on the answers
30:13
which are given there is a result so every token so now this friend
30:19
right it will have some values for each of these questions maybe the values are .1 2.1.3 Etc and if we are assuming that
30:29
there are 768 questions this will now be a vector of 768
30:36
values one thing which I want to point out here is that this number of questions here 768 that vary from one
30:43
large language model to another large language model so now if you go and and see let's see for gpt2
30:50
gpt2 token embedding Dimension so if you search gpt2 token
30:57
embedding Dimension we'll see that it's 768 right uh but here also gp22 small
31:02
had 768 but the largest gpt2 had 1600 Dimensions so this number of questions
31:09
768 actually varies from one large language model to another so here what we are going to do
31:16
is that we are going to assume that the number of questions is 768 for gpt2 but
31:21
remember that if the token goes to different llms it might be asked different questions so now imagine you
31:28
are a token you have been given a badge or a role number and suddenly you are asked this huge set of 768 questions you
31:35
respond then your answers are collected in one 768 dimensional
31:40
Vector that is called as the token embedding Vector so now along with the badge you also carry your result with
31:47
you so you have a badge with you and you have this result of 768 values with you
31:52
that's what has happened to you until this stage right that's the stage of token embedding the difference again between
31:59
token ID is that token ID does not carry any notion about semantics whereas token
32:04
embedding in token embedding assignment we care a lot about the meaning of the word itself the reason token embedding is
32:11
done is that to create llms you ultimately need to extract meaning right you're teaching something about the
32:17
language to the model so this is a very crucial step these set of questions or these uh set of features which are
32:26
collected about every single token so until now every token has a badge and every token has 768 value result sheet
32:35
which they take along with them then one more thing which also matters is your position among your
32:41
neighbors so here if you see a true friend accepts you so friend comes in
32:47
the middle of the sentence right it comes at position number three over here so a comes at position number one
32:55
true comes at position two friend comes at position number three accepts comes at position number four and youu comes
33:02
at position number five so the friend is coming at position number three and that position also
33:08
matters why does the position matter because if you say the
33:15
dog the dog chased another dog okay if you take a look at this
33:23
sentence you need to somehow be aware that this dog is basically different than this second dog so if you just take
33:30
the meanings of the words right as in Phase number three we just took the meanings so the token embedding for this
33:37
dog and this dog will be the same but actually there there are two separate dogs and we need to teach the model
33:43
related to that so the only way to distinguish between this dog and this dog is to know that this comes at
33:49
position number two and this comes at position number five so as a result it's important to also have some knowledge
33:57
about the position so similar to the 768 questions which we
34:02
asked 768 questions will again be asked with respect to the position so remember
34:08
that although this number varies across different models if you fix a particular
34:14
language model the number of questions which are asked in token embedding and
34:19
the number of questions which are asked in the positional embeddings are the same so if we are looking at gpt2 small
34:26
right now as the model there were 768 questions asked in token embedding
34:31
similarly 768 questions will be asked in positional embedding and what might these positions
34:39
or what might these questions be they might be something like are you at the beginning or are you around the middle
34:44
of the sequence or do you encode long range dependencies Etc actually no one
34:50
knows what these questions might be but I thought this the simplest way to explain about positional embeddings and
34:57
token embeddings so now in Phase number three every token had a token embedding which
35:03
is associated with them that's a 768 dimensional vector and when you come to phase number four based on the position
35:09
you are asked these 768 questions right so you will also have a 768 dimensional position embedding which is associated
35:16
with you so now imagine what all a token is subjected to a token first has a
35:22
stamp um or a badge of token ID then it has the token embedding result those are
35:28
the questions which it needs to answer that's the first test then the token goes to another test which is positional
35:33
embedding and then it again has this 768 values it's a lot of processing which
35:39
needs to be done for every token it essentially has to go through a huge number of tests and then in Step number
35:46
five what we do is that we add the result of your token embeddings plus the positional embeddings so you don't have
35:52
to carry these two results separately anymore you merge both of them so the token embedding which is now 768
35:59
dimensional vector and the position embedding which is now 768 dimensional Vector is added together and that is
36:07
what is called as input embedding so this is the input embedding
36:12
for the token which is frint that's a 768 dimensional Vector we'll have similar input
36:18
embeddings for all the other tokens or all the other words but here I showing to you the input embedding for the token
36:24
which is sprinted that's the result of the token embedding plus the positional embedding so now you don't have to carry
36:30
these two results separately you just carry one result that's now the
36:36
768 dimensional Vector which is associated with u as a token that is the most important
36:43
distinguishing Factor you carry with you in the rest of the journey now think of
36:48
the journey right you are first isolated you are given the badge you're given the first test of token embedding second
36:54
test of token embedding finally after going through all of this steps you now
36:59
have one thing to distinguish you and that's the input embedding so you can
37:04
think of this as your uniform now this is a uniform which is specially created for you and you wear
37:10
that uniform as a token every friend or every token along
37:16
with you every other word along with you will wear a separate uniform why will they wear a separate uniform because
37:23
their meanings would be different so they would answer these questions differently their position positions would be different so they would answer
37:29
these questions separately so the uniform for every token will be
37:34
different and until this stage what we have done until now these five steps
37:39
which we saw are essentially what is happening in the input block or the
37:45
input layer so that's the first part which we have studied until now uh and
37:50
now I think these three steps which have been mentioned here will be very easy for you to understand the first is the
37:56
tokenization the second is the token embedding the third is the positional embedding that's exactly what we saw
38:03
right and then the token embedding and the positional embedding are added together to give something which is called as the input
38:09
embedding that's all which happens and remember the tokenized text here we saw
38:16
the token ID assignment through the vocabulary or through the book of token
38:21
IDs so that's the input embedding after every token goes through the input block
38:28
which is Phase part number one over here they have a uniform which distinguishes them from the other tokens all right so
38:35
that's the first part which is the input then once you have an uniform only then
38:41
you are ready to go to the next part which is the processor so as a token you have a
38:47
uniform now and now you're ready to unboard the train to the Transformer block so it's similar to Harry Potter
38:54
let's say where you can go to the school only if you have let's say a certain uniform you you belong to a certain
39:00
house let's say Gryffindor or Raven Club Slytherin Etc so every word or every
39:07
token now has a uniform and now you're are finally ready to onboard the train to the Transformer block so here you see
39:14
these five will be sitting together in the Transformer block a true friend accepts you and whenever I show uniform
39:21
right now you should think that the uniform means a 768 dimensional Vector every token now so the trans for the
39:28
Transformer block it does not understand words it does not even understand the meanings of words all currently all it
39:36
essentially knows is that every token is a 7608 dimensional Vector some magical
39:42
things will happen in the Transformer block so that the meanings between different tokens will be understood very clearly and the model itself will learn
39:49
about the language so the Transformer block essentially is where all the magic
39:55
happens the second processor part is where really everything is happening
40:01
this this part over here this processor part is really where all the magic happens and how so you might be thinking
40:09
that how do llms work so well um they they almost interact with me as a
40:14
interact with me as a human although they predict the next token they seem to have learned something about language
40:20
itself they summarize tasks they are good at grammar checking they draft emails for me uh they do complex coding
40:27
for me all that is because of what is happening in the Transformer block so now every token essentially
40:35
goes on a journey through the to the Transformer block itself okay now to think of the
40:42
Transformer block we need to understand that the Transformer block is like a train with a huge number of different
40:49
components right so first we are going to look at the components of the Transformer block train itself and I'm
40:55
not going to go through all the compon components in detail I'm just going to briefly explain to you what each
41:00
component of the Transformer block does so now imagine that these five
41:06
passengers have been assigned a compartment right and all of them are now input embedding 768 Dimensions they
41:13
have to go through 1 2 3 4 5 six steps within one Transformer block what are
41:20
these six steps so you can think if you think of a transformer block as a train these six steps can be thought of as six
41:26
compartment ments which are connected together so once you join the train you have to go with your neighbors and you
41:33
have to go through all these six steps the first step is essentially layer normalization which means that the 768
41:40
dimensional vectors so let me now just focus on friend the 768 dimensional
41:45
Vector for friend is normalized which means its mean and standard deviation are adjusted so that mean becomes zero
41:51
and standard deviation becomes one that step is easy then we come to multi head
41:57
attention so here you see I have marked this with a different color because this is truly The Innovation which Powers the
42:05
Transformer Block in multi-head attention we essentially learn if we look at one token how much attention
42:11
should be given to other tokens so if you look at friend how much attention should be given to a true accepts and U
42:21
so multi-ad attention essentially encodes something about the context if you look at one token you suddenly make
42:27
a map of how important are all the other tokens and if you think about it that helps a lot in understanding things
42:34
about the language in understanding context of a sentence itself or understanding context of a paragraph So
42:41
if I say something like uh I am from Pune India I
42:49
speak so here here if you have to complete the next sentence you need to know that you need to pay more attention
42:56
to Pune and India because that is where I from right so you need and not pay too
43:02
much attention to the first three tokens maybe so that is why attention mechanism is important to understand the context
43:09
of a sentence and to predict the next token we are going to learn about the attention mechanism uh in a lot more
43:16
detail in the next lecture but remember that that's the second compartment of the Transformer block the third
43:23
compartment of the Transformer block is the Dropout layer uh if you have learned about neural networks Dropout is
43:29
essentially if there are 100 parameters and Dropout factor is 05 you randomly turn 50 of them zero why because what if
43:38
some parameters are lazy and they're not learning anything at all suddenly if the other parameters are
43:46
now dropped out which means they're set to zero these parameters have no option but to learn something on their own so
43:53
Dropout is a mechanism to get lazy parameters back into action it improves the generalization performance and it
44:00
prevents overfitting so there are two layers of Dropout if you see there is one layer of Dropout here and then there
44:06
is one layer of Dropout again uh after the Dropout layer we have a skip connection or a shortcut
44:12
connection U shortcut connections just help the gradient to flow through an alternate path and they make sure that
44:19
we don't have the vanishing gradient problem then so after we go through
44:24
normalization then multi-ad attention drop up after that we have another normalization layer which does the same
44:32
function as the first one then we have a feed forward neural network this is again a very important
44:38
component of the Transformer block itself this feed forward neural network if you think about it so there are 768
44:46
Dimensions right in friend which is my token right now the feed forward neural
44:51
network essentially takes it into a much higher dimensional space which is 4 * 760
44:58
and then it compresses it back into a 768 dimensional space so this expansion contraction make
45:05
sure that we are exploring more richer spaces we are exploring spaces which have more dimensions and more parameters
45:11
so that just makes sure that the Lang our language models have enough parameters to capture additional
45:18
complexity this speed forward neural network is where the mixture of experts innovation has actually happened for
45:24
deep seek finally we have another another Dropout layer and then we have another
45:30
skip connection or a shortcut connection remember these plus signs wherever they are there they resemble skip or shortcut
45:36
connections and they just make sure that the gradient has alternative routes to flow because if the gradient flows in a
45:43
chained Manner and if the one gradient is small once it's multiplied together the gradient will become zero right or
45:51
if the gradient is large and if it's multiplied together it will blow up so it can lead to Vanishing gradient
45:57
problem where the learning will stop or it can lead to the exploding gradient problem where the learning will be very
46:05
unstable uh so this is how these these five tokens have to go through these
46:11
different steps every token has to go through normalization multi-ad attention Dropout skip connection normalization
46:18
again feed forward neural network again Dropout layer again and then one more skip connection so that's the
46:25
Transformer block and if you see this schematic which we saw at the beginning
46:30
of the lecture you'll see that the same thing has been mentioned over here every layer has to every token has to go
46:36
through a layer normalization uh attention Dropout then skip connection again a layer
46:43
normalization of feed forward neural network Dropout and a skip connection so that's the journey which
46:49
has to be followed through a token this seems like such a tedious journey to follow right uh I have first of all I
46:56
have to go through all these five steps to get my uniform and then on top of that after that I have to go through
47:02
every block every Transformer block and go through these steps but there is one more additional layer of complexity that
47:10
just like this one Transformer block one large language model has multiple Transformer blocks right so if I say
47:19
gpt2 how many Transformer blocks does
47:24
gpt2 have so if you take a look at gpt2 itself
47:30
gpt2 small has 12 Transformer blocks gpt2 medium has 24 gpt2 large has 36
47:36
Transformer blocks and gpt2 Xcel has 48 Transformer blocks so even if we look at
47:41
the small one right now each Transformer block has all of these steps so now
47:47
every token has to essentially do all of these steps 12 times uh so that so
47:54
that's why these if you think of one Transformer block as one part of the train there are 12 such Transformer
48:00
blocks which are linked together and one token has to essentially go through all of these 12
48:06
Transformer blocks so the journey is extremely more tedious so here I have
48:11
mapped it out right these are 12 Transformer blocks which essentially every token has to go through uh so the
48:18
token friend there right it has to go through the first Transformer block it has to go through
48:24
the second it has to go through the third see similarly it has to go through all of these 12 Transformer blocks so
48:30
number 12 over here there is a very tedious train journey which every tokon has to essentially follow getting a
48:37
uniform is a struggle we have to go through five steps going through the processor is an even more struggle
48:43
because you have to go through all of these 12 steps again um so this is what is actually
48:50
happening in the processor which is the part number two which we have just seen in the processor what happens is that
48:57
so here I have shown one Transformer block rate you can think of this multiplied by 12 if we are using GPT
49:02
small if we are using the largest gpt2 then it's 48 Transformer blocks and
49:08
modern gpts might have 96 or even more Transformer blocks so every token has to
49:13
now go through all of these and remember that the dimensions of a token are usually maintained even when it comes
49:20
out of the Transformer so let's say the input right the input uniform which we saw was 768 dimension here if you
49:28
remember it was a 768 dimensional Vector right here 768 after going through all
49:34
of these Transformer blocks after going through the 12 Transformer blocks it comes out of these 12 Transformer blocks
49:41
by retaining its Dimensions so it still has 768 Dimensions so now a true friend
49:47
accepts you have come out of the Transformer block and all of them still have 768 Dimensions naturally the values
49:54
have been changed right and then they go through now we go to
49:59
the output layer there is one step of normalization here so if you see over
50:06
here there is one step of normalization here which is called as the final layer
50:11
normalization so that step of normalization is mentioned over
50:16
here so every 768 dimensional Vector again goes through this stage of
50:22
normalization and then we have one last layer which is very important so remember now that we have reached the
50:28
last layer we have uh a true so this is
50:33
a true friend needs you and each of these is a
50:39
760 a dimensional Vector right now we have to somehow convert the
50:45
768 Dimensions into our vocabulary size which is 50,000 because now we have to
50:50
predict the next token so then every token is essentially passed through a
50:55
neural network whose size is 50,000 or size is uh 768 multiplied
51:04
50,000 so that when these vectors are multiplied with this we result into
51:10
50,000 dimensional vectors for each token so now the uh size for each token
51:16
is that a true friend needs
51:22
you right the size for each token now is going to be 50 ,000 after it passes
51:27
through the output layer this layer is also called as the output projection
51:33
layer and after every token goes through the output projection layer it has a dimension equal to the vocabulary size
51:40
so remember our vocabulary size is equal to 50,000 so this 50,000 is coming from the
51:46
vocabulary size and I'm just going to explain why do we need its Dimensions to be equal to the vocabulary size and the
51:52
final last step is choosing the next token right so now once we have reached the last step what do we have we have
51:59
five tokens a true friend accepts you and for each of these we have a 50,000 dimensional Vector now what we are going
52:06
to do is that we are going to look at these 50,000 Dimensions we are going to look at that index which has the highest
52:12
value or the highest probability then we are going to find that index over here and then we are going to look for its
52:19
corresponding token that's it so if the index here is so for the
52:24
first uh uh the so a true friend accepts you right
52:30
there are multiple input output tasks here when a is the input true should be the output when a true is the input
52:36
friend should be the output when a true friend is the input accept should be the output when a true friend accepts is the
52:43
input you is the output when a true friend accepts you is the input something else will be the output which
52:48
is for so a true friend accepts you if you look at this sentence and you have to
52:54
predict the next token it's not not just one next token prediction tasks there are multiple input output tasks within
53:01
this same sentence what are these input output prediction tasks when a is the input true should be the output Etc and
53:08
only the final thing here is relevant for us which is the next token
53:14
prediction initially we'll of course not get good tokens right but we'll have the loss
53:20
function which is based on the actual values so this is the actual next token which I want but the predicted ones
53:27
initially will be completely far off and that's when back propagation comes into
53:32
the picture when all the parameters which are there they are actually optimized we'll come to that in a moment
53:38
but for now just let me explain the final step again we have every token and
53:44
now every token is associated with a different uniform let's say whose dimensions are
53:50
50,000 why do we have Dimensions 50,000 because we have to predict the next token for every word which is is over
53:56
here so for o we have to predict the next token so we look at that index which is the highest or that token
54:03
ID which has the highest probability here we go to the book of the words or book of token IDs and then we reverse
54:11
map the word which is corresponding to that token ID so if this token ID here is let's say 555 or this token ID is
54:19
5,000 I go to the word here which is 5,000 and I ideally want true over here
54:26
but initially when things are not train maybe have four so the actual prediction
54:31
might be four similarly I'll get the actual predictions for True maybe the highest
54:37
token ID is here friend here and then I'll get the highest token ID here and
54:44
that's how I'll predict the next token for each of these and then I'll find the loss term between the actual value and
54:50
the predicted value that's how the entire architecture of the llm is structured so now if if
54:56
you go to the output layer which is my final layer here you see we have two things
55:01
which are changed to each other right we have the final layer normalization layer which is connected to the output layer
55:08
and then we have the Matrix for next token prediction this logits Matrix is this this logits Matrix is this this one
55:16
which I showed to you right now and we use that to make the next token prediction so now you might be thinking
55:23
that what are all the parameters which are optimized here so right from the start itself these token embedding
55:29
values which are there we do not know them a prior so let me Mark the parameters which are trained by a star
55:36
these we do not know a priority so these are trained positional embedding assign assignment we do not know a prior so
55:42
these are trained then uh every single aspect of the Transformer block has some
55:48
parameters multi-ad attention as parameters that is trained uh the feed forward neural network has parameters
55:54
that is trained and there are are 12 or 24 such such blocks that even increases
55:59
the parameter size further um so there are a huge number of
56:05
parameters throughout this entire process which are trained even this final neural network it has these many
56:11
parameters which need to be trained all these parameters add up together to lead to the total number of parameters which
56:18
are 175 billion or maybe a trillion so think about the engine which we started
56:24
out right we started out with knowing that we started out with thinking okay this is the llm engine but how does the
56:31
llm engine actually work what are the parameters beneath the llm engine and
56:36
where are these 175 parameters actually going 175 billion so now we have taken a
56:41
look at the detailed architecture which is the input which is the processor and which is the output so and we have seen
56:48
the Journey of a single token right a token essentially first goes through the input phase which it is isolated it is
56:55
assigned a token ID or a badge then it's given one quiz or one set of 768
57:00
questions that's the token embedding it encodes meaning then it's given a second set of questions which is positional
57:07
embedding that encodes its positional value we add the token and the positional embedding that gives the
57:12
input embedding or the uniform for every token with this uniform different tokens sit on the train to the Transformer
57:20
block and each Transformer block essentially has the normalization layer
57:25
multi-ad attention drop out normalization again feed forward and Dropout interpers with two skip
57:32
connections and there are 12 such blocks like this in gpt2 in gpt2 XEL there are
57:39
I think 48 such blocks but in the advanced gpts there might be 96 or even more number of blocks like this so every
57:46
token needs to go through all of these blocks when it comes out of all these blocks it size Still Remains 768 then it
57:53
goes through one more normalization layer size is 768 and then finally we have an output layer where for every
58:00
token which we have it's converted into a vector now with size of 50,000 which
58:06
is equal to the vocabulary size and then we look at every we we look at every
58:11
token basically it's 50,000 dimensional vector and then we look at that token ID
58:16
which is the highest probability and we use that to predict the next token so in
58:22
one sequence which we have we have multiple input output prediction tasks so if we have a sequence with five
58:28
tokens there are five input output prediction tasks which essentially give our loss function and then our loss
58:35
function is basically back propagated and all the parameters are optimized all these 175 billion parameters which come
58:42
through in several stages in token embedding there are parameters in positional embedding there
58:48
are parameters then in U several aspects of the Transformer block there are
58:53
parameters in the output layer there are parameters all of these parameters are essentially optimized through back
59:01
propagation and then ultimately what we have is a model which has intuition about language itself and it can also
59:08
predict the next token so next token prediction is the task as you see here we are predicting the next token and we
59:14
are comparing with the actual value that's the task but in this task since we have so many parameters the byproduct
59:20
is learning the language itself so in today's lecture my main
59:25
purpose was to take take you through the Journey of a token think from the point of view of what happens with one token
59:32
try to open this engine try to open this engine of the llm and really try to see how the engine is actually working and I
59:39
hope I have conveyed that to you the reason I constructed this analogy or a story of a journey of a token is for you
59:45
to really understand what goes on inside the llm architecture because without understanding that uh we cannot move
59:52
ahead to the next part which is attention now the plan is that in the
59:58
next lecture I'm going to motivate why we need attention in the first place then we are going to look at self
1:00:03
attention then we are going to look at multihead attention then we'll look at key value cache so if you see the next
1:00:10
plan is the need for an attention mechanism then we have self attention and then ultimately we have the
1:00:15
multi-head attention mechanism so all the future lectures are planned in a lot of detail as I mentioned this won't be a
1:00:21
small series with five or 10 minute videos every single video of this series
1:00:27
will be pretty long around 40 to 45 minutes and I will plan to go through the entire steps so multi-ad latent
1:00:33
attention is a very important concept but I want all of us to be at the same page when we actually understand that
1:00:39
concept thanks a lot everyone and I really look forward to seeing you in the next lecture please make notes along
1:00:45
with me this series can get a bit difficult I'm trying to distill the concepts in as easy to understand a
1:00:52
manner as possible but still there might be some challenges across the way so please make note so that you strengthen
1:00:58
your Concepts thank you everyone and I look forward to seeing you in the next lecture



