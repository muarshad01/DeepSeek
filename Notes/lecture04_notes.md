#### The need for attention mechanism

* Here is how the field of Generative AI has evolved

| Model | Year|
|---|---|
| RNN | 1980 |
| LSTM | 1997 |
| Attention | 2014 |


***

* [ChatGPT](https://chatgpt.com/)
  * I want to learn about AI. Can you help me?

***

* 10:00

* NN can't deal with memory.
* Context

***

* 15:00


pass it to the second then I decode the second token and then I decode the token this is how language to language
16:02
translation Works in Long in recurrent neural networks lstms are a bit
16:07
different to predict the current hidden State based on the previous hidden State they also maintain something called as a
16:13
cell state so lstms have a provision for taking into account long memories as
16:19
well as short memories so that's essentially to increase the context window they deal with but the
16:26
architecture fundamentally Remains the Same it just that the the mathematics beneath is a lot more complicated for
16:32
lstms now do you see a problem with this take take a look at this example
16:40
which we had for discussing about context right what is the problem with recur neural networks when let's say you
16:46
have big paragraphs to deal with let's say you you have a big paragraph and you want to translate that entire paragraph
16:53
what is the problem with recurrent neural networks in doing that
16:59
so let's take a simple example so that I can explain that problem to you a bit better imagine that you have this
17:06
paragraph okay I have constructed this randomly from chat GPT I want you to do the following I
17:12
want you to first read this paragraph once maybe take 30 seconds or 1 minute or more to read this paragraph Just Once
17:19
then close your eyes and then translate it into any language you want you can translate it into Hindi Spanish French
17:27
German whatever language you speak other than English so close your eyes and
17:32
translate it into another language where you able to do this it's
17:40
impossible right because when you close your eyes you essentially probably remember some summary of what you have
17:45
read you don't remember exactly each and every word but to translate we have to
17:51
look at each and every word and translate it individually right that's the exact same problem which is
17:56
happening with recurrent neural networks why because we are relying on only one
18:01
hidden state or one vector to capture all the context of what has happened in the past so it's like closing your eyes
18:09
and then trying to remember the entire paragraph there is so much pressure on this one hidden state or this Vector to
18:16
recollect all the memory that if it's a huge paragraph how can you compress all
18:22
of that information in a 500 dimensional vector or a thousand dimensional Vector
18:27
naturally some information is going to get lost and remember the decoder does not
18:33
take input from the first hidden State or the second hidden state it only takes the final hidden State as the input
18:40
that's where the main issue arises the context which goes from the encoder to the decoder is only from the final
18:46
hidden state that is why this is called as the context bottleneck problem the final hidden state is like
18:53
you when you closed your eyes and when you tried to memorize the entire paragraph it's impossible to do that you
19:00
cannot put so much pressure on one vector so context bottleneck in
19:07
recurrent neural networks is really not good for retaining a long range context and this serves as a great pathway for
19:15
understanding why attention comes into the picture so now if you look at the recurrent neural network let me rub some
19:21
things over here or erase some things what do you think is the ideal solution to this problem if let's say this is the
19:29
problem uh which we are encountering right all the context cannot be captured in one single Vector uh what is the
19:37
solution to this problem well the solution seems to be that let's say when we are decoding when
19:44
when I'm decoding this uh let's say when I'm decoding from the first hidden state
19:49
of the decoder instead of having access to just the final hidden State what if I
19:54
have access to all the hidden states of the encoder and all the inputs
20:01
also so if during decoding I have access to all of the inputs and then I can make
20:08
predictions such that okay if I'm decoding the first token I should give more importance to
20:14
the first hidden State and I should not give importance to the second hidden State and I should not give importance
20:20
to the third hidden State what if I could create a mechanism which
20:26
can understand the Rel relative importance which needs to be given to different tokens this is the most
20:33
important part of this lecture so please try to focus our here the main thing
20:38
which can solve the context bottleneck problem is that when I'm decoding what if I develop a
20:45
mechanism which can try to quantify how much importance I should give to every
20:51
hidden state so while the first decoding is taking place let's say I give 80%
20:57
importance to the first hidden State 10% importance to the second inter state and 10% importance to the third inter state
21:04
if I have this information along with me then the context bottleneck problem is solved right because even if the
21:10
paragraph is long if I can attend to all the tokens in that paragraph I can essentially also give importance to
21:17
something which has come in the past so let's say if this is the sentence and I'm predicting something
21:23
here in the attention mechanism I can just try to see how much attention to give to the first token the second uh
21:31
the second token third and the entire paragraph and then I can say that I need to give maximum attention to these
21:37
tokens over here so you see I'm already starting to use
21:43
the word attention over here you can think of it as relative importance to
21:48
individual tokens in the paragraph So the ideal solution is that
21:55
we need to selectively access parts of the sequence during decoding so when you
22:01
are decoding let's say I want to make certain predictions so that if I'm decoding from the if I'm decoding the
22:07
first token right uh this alpha 1 one is the relative importance given to the
22:13
first hidden State Alpha 2 one is the relative importance given to the second hidden State Alpha 31 is the relative
22:20
importance which is given to the third hidden state so I want to be able to make predictions such that Alpha 1 one
22:26
is 100% Alpha 2 2 1 is z and Alpha 31 is zero something like that but you see
22:32
here instead of just relying on H3 I'm now relying on all of my previous hidden
22:38
States and not just that I'm quantifying how much I should rely on each of the
22:43
previous hidden States so again this is important we need to selectively access parts of the
22:49
input sequence during decoding and I want to explain this to you with this
22:54
example again let's say I take a screenshot here okay um and then I put it over
23:01
here let's try to understand what the statement actually means so if I ask you
23:07
to translate this paragraph How would you actually do it you would probably start um you would actually probably
23:14
start with let's say this right let's say you start with this that how much words you can look at at one point let's
23:21
say that's your context window you cannot look at more words than this at one point then what your mind will do is
23:28
that for your mind the rest of what comes in the paragraph is irrelevant right so you will mask out all of that
23:35
in your mind you will mask out all of that it's not relevant so what you will do is that all of these other things are
23:43
not important to you now you just want to look at the current
23:49
context and you want to selectively hide out every other thing right so what your
23:54
mind does is that your mind pays attention this is very important your mind is paying attention to this part of
24:02
the paragraph and then it's starting to translate only this part of the paragraph and after this is translated
24:09
then your mind will move to the next maybe next five words uh after this is translated your
24:16
mind will move to the next five words and then you will go to the next five words Etc so you see what you're doing
24:22
here at a particular time your mind is making a decision that I want to pay the maximum attention to these token
24:30
when you're looking at the first uh first context window and then your mind
24:36
is also making uh certain quantitative judgments that don't pay any attention to the rest
24:43
of the tokens at all don't pay any attention to these so
24:49
that's what is actually meant by selectively access parts of the input sequence during decoding right
24:57
selectively accessing parts of the input sequence that's the most important sentence to understand the attention
25:03
mechanism it means that only pay attention to that part of the input
25:09
sequence which is important at that
25:15
moment and this seems like an intuitive thing to do right but it's actually the
25:22
most important thing which helped language models to become better so imagine that uh
25:29
you want to find a spelling mistake in a piece of paragraph right what do you do
25:36
you want to find out where the spelling mistake is you go through the paragraph step by step so you pay attention at one
25:41
then you pay attention to the next Etc this is what llms also need to do they
25:48
need to selectively pay attention to the part of the sequence and then find that part let's say which has the spelling
25:54
mistake for example right so this is the main purpose of why we need attention while
26:02
decoding we can quantify how much atten importance or attention needs to be
26:08
given to each input token that is what I'm getting at at the moment now the first paper which really
26:15
implemented this was not the attention is all you need so if you see this paper
26:22
this paper has around 1 Point 170,000 citations right
26:29
huge number of citations so many people actually believe that the um many people
26:35
believe that the attention mechanism was first introduced in this
26:42
paper but the first paper which actually introduced the attention mechanism is
26:48
this bhano attention mechanism so if you search bhano attention and uh you click on this
26:55
you'll see that this is the first paper which actually implemented the attention mechanism for translation tasks this
27:01
paper I think came out in 2014 and this was published at iclr in 2015 their main purpose was they
27:09
essentially implemented this architecture in which we can selectively decide how much attention to pay to each
27:16
hidden state so that's what's called a A1 A2 A3 that's the attention score and
27:21
based on that you can do the translation tasks and uh they essentially proved
27:27
that we can do sequence to sequence translation in a much better manner if we use the attention
27:35
mechanism um so just keep this particular paper in mind so here you can see that on the on this year we have the
27:43
English sentences and on the y- axis we have the French translation and the
27:48
diagonals are essentially showing the English word paying the most attention to the translated French word and what's
27:56
cool thing about this is that the order is not the same so the agreement on the
28:01
European economic area right European economic area that's English but the way
28:06
it translates to French is economic European right so it translates to area
28:12
economic European so it's not direct work toward translation and still the
28:18
attention mechanism essentially figures out that see this is the maximum uh this is brightest right which
28:24
means the European year in French corresponds to the Eur European year in English although they both don't occur
28:31
at the same position so the European in English occurs at position number five and the European in French occurs at
28:38
position number seven so all the max all the brightest areas are not on the
28:43
diagonal there are some which are of the diagonal itself U and even that is captured by
28:50
the attention mechanism so the coolest thing about this plot is that this off diagonal elements because translation
28:56
does not always happen word by word right there are some words which come before in some languages which are which
29:02
come after in some other languages based on how nouns verbs Etc are arranged so that's the cool thing about this paper
29:08
this was the first one where the attention mechanism was used for translation task um and
29:16
then attention is all you need paper is where the Transformer block was introduced and the attention mechanism
29:23
was integrated within the Transformer block that's the main uh advantage or
29:28
that's the main unique point of this paper attention is all you need right so what happened is that RNN
29:36
and lstms are quite good for language translation but they have this context botanic problem uh so researchers
29:43
figured out that RNN architectures are not really needed for building deep neural network so 2014 is where the
29:51
badano attention mechanism was introduced where they had the RNN so they retained the RNN plus they had the
29:58
attention mechanism so essentially in bhano attention you can think of the encoder
30:03
decoder Block in a similar manner but just they added this attention mechanism
30:09
uh 3 years later which means in 2017 researchers essentially figured out
30:14
that RNN architectures are not even needed for building deep neural networks for
30:20
NLP um and they propos the Transformer architecture so think about the field
30:26
which evolved in this way right 1980s was RNN 1997 was
30:36
lsdm actually 1966 was Eliza let me write this also then 2014 was where the
30:44
attention mechanism was introduced but that was still attached to RNN the encoder decoder architecture of RNN 2017
30:52
is when researchers figured out that rnns are not needed for natural language
30:58
processing task so RNN were scrapped out and then essentially the attention mechanism remained but it was coupled
31:04
with the Transformer block that's the main difference between
31:10
the 2014 paper and the 2017 paper in the 2014 paper the RNN block was still there
31:16
but that was then removed and then that was replaced with the Transformer block or the Transformer architecture in 2017
31:24
and then it led to this whole architecture right we now have the Transformer block and within that we
31:30
have the attention right so this is the GPT architecture this is not the architecture in the original Transformer
31:37
paper the original Transformer paper had an encoder as well as the decoder whereas this architecture which I'm
31:43
showing you right now only as a decoder don't get confused by this at the moment the main purpose of this today's lecture
31:49
is for you to understand why attention needed to be introduced in the
31:55
historical perspective of natural language processing and the main reason is that the one
32:00
sentence you need to keep in mind is that we need to selectively access parts of the input sequence during decoding
32:07
and first attention was introduced in RNN itself then researchers figured out that
32:13
okay let me get rid of the RNN still let me try to merge the attention mechanism
32:20
somewhere and that's when they merged it with the Transformer block over here that was in 2017 then in 2018 is when
32:29
uh the GPT architecture came out so then it's the attention plus GPT so GPT is
32:35
based on this original Transformer architecture but instead of having the encoder and the decoder block it only
32:40
has the decoder block as you see over here and it retains the attention mechanism over here so the thing which
32:48
we talked about why attention mechanism is needed it manifests itself over here in this part of the llm
32:55
architecture now we are going to start looking at at uh if you take a look at next token
33:01
prediction tasks what is self attention really and uh what are context vectors
33:09
and what is the main purpose of the attention mechanism so let's uh what is the main purpose of the self attention
33:15
mechanism for next token prediction so let's start learning about that now now
33:21
that we have understood about the attention mechanism and the history of the attention mechanism let's try to
33:27
look at at this term self attention what does self attention actually mean so
33:32
self attention means that it's a mechanism which allows every position in the input sequence to attend to all the
33:39
positions in the same sequence what this means is that let's say if I have a sentence right um if I have a sentence
33:46
such as the next day is
33:55
bright self attention essentially means that until now if you looked at the RNN
34:00
we saw the attention which needs to be given from the decoder to the encoder right so if the first decoded word is a
34:07
French word uh we are basically looking between two different sequences so the
34:14
English sequence is this let's say the English sequence is this I will eat and the French sequence is
34:24
this uh this is the French sequence and we are looking at if you are doing the first decoding how much attention should
34:31
you give to the English sequence so here the attention is between sequences it is
34:37
not within the same sequence self attention is on the other hand when you're predicting the next token which
34:43
is typically done for the llm since llms predict the next token right they're not
34:49
specifically trained for translation tasks to predict the next token you essentially don't have different
34:55
languages you just have a bunch of data right so instead of having attention
35:02
between two sequences what you do is that you just take the same sentence and let's say if you take a look at
35:08
next you try to find out if you look at one token or one word how much attention
35:13
should I pay to all the other tokens in the sentence that's the most important thing
35:20
to understand over here if you look at one uh token essentially or one word
35:29
then how important are the neighboring words to that particular token why is
35:36
this knowledge uh can you try to think why this knowledge is important for us
35:43
why do we need to encode the knowledge of the tokens which essentially
35:50
surround uh a given token the reason this knowledge is
35:56
important to us is because because when you are predicting the next token you essentially need information about the
36:03
context of a sequence you need information about how different words
36:08
relate to each other so again taking the same example right let's say if I say
36:14
that I am from Pune
36:19
India I speak let's say this is the sentence right if you look at
36:25
speak I need to know that I when I look at speak the maximum
36:30
attention needs to be paid to Pune and India maybe all the other words are not as relevant because my dialect what I
36:39
speak is very heavily influenced with the region which I come from so when you look at a cloud of
36:46
words and if your Transformer architecture or if your llm engine has
36:53
information about how one word relates to other words which are surrounding it
36:59
and how much importance needs to be paid to the different word surrounding it then it just becomes very very easy to
37:06
predict the next token right if you look at uh one if you
37:11
look at one token so that's the reason why um the self attention mechanism
37:19
becomes very important if self attention mechanism was not there you would have lost this contextual information about
37:26
how other tokens relate to a given token which we have chosen and I hope now you
37:31
understand why it is called self attention in this case when we are looking at sequence to sequence language
37:36
translation the attention is between sequences but self attention is when we
37:42
look at one sentence itself and we look at tokens within the sentence and we essentially see how these tokens
37:48
are relating to each other right okay so
37:54
let's take the same example again the next day is bright so the next day is bright and remember that when these
38:00
tokens go to the Transformer architecture they are now vectors as you have seen before they have this uniform
38:07
now remember every token has a uniform which was a 768 dimensional Vector here
38:12
that's the input embedding right so whenever I'm showing these these blocks here it essentially means a vector so
38:19
the next day is bright these are all vectors now for a Transformer it does
38:24
not understand words it does not understand sentences also all it knows is that every token is a vector so the
38:31
is a vector which I'm calling X1 next is a vector which I'm calling X2 day is a vector which I'm calling X3 is is a
38:39
vector which I'm calling X4 bright is a vector which I'm calling X5 right and now if I'm looking at a specific word
38:46
let's say next as I showed over here I want to see if I look at this word next
38:52
how much attention should I give to all the other tokens and this this attention is given
38:58
by Alpha 21 or I'm calling it a symbol Alpha 21 why 21 because next is the
39:04
second token and I'm wanting to find the attention score between the second token and the first token that's Alpha 21 this
39:11
will be Alpha 22 this will be Alpha 23 here will be Alpha 24 and here will be
39:16
Alpha 25 I essentially want to find out all the attention scores if I'm looking
39:21
at particular token so this is called as a query the token which I'm focusing on
39:26
right now that's called as the query query token and uh I want to find out if
39:31
I'm looking at the query how much attention should I give to all the other uh tokens these are also called as Keys
39:40
sometimes in the common nomenclature right uh so ultimately what I want to do
39:47
I want to take this information let's say I get these attention scores I want to somehow take all of this information
39:53
and transform this Vector from an input embedding Vector to a context vector
39:58
now here is very important distinction which I want to make currently next is a input embedding Vector right so it
40:04
contains token embedding plus positional embedding but context Vector is something very different so if this is
40:11
the input embedding Vector for next um and if I plot the context Vector
40:18
in the same space here so this is the context embedding Vector for next the
40:23
context Vector is actually much richer than the token embedding vector why because the token embedding vector or
40:30
the input embedding Vector contains no information of the neighboring words but now my context Vector consists of
40:37
information of my neighbors also that information is now baked into my input
40:43
embedding so if you have an input embedding if you have the input embedding Vector which is the uniform
40:50
which I talked about right and if you augment this input embedding vectors with context about the neighbors
41:00
we'll see how this augmentation is done but essentially this leads to something which is called as the context
41:08
Vector so the whole goal of the attention mechanism or the self attention mechanism is to convert all
41:15
the input embedding vectors to context vectors so all of these uniforms so we
41:21
saw U these uniforms right all tokens have a 768 dimensional uniform where
41:27
when they come out of the normalization layer and when we go to the multi-ad attention layer what comes in the
41:33
attention layer is an input embedding Vector what comes out of the attention layer is a context Vector so something
41:40
much richer comes out after we exit the attention block and that's why I marked
41:45
it with a different color the reason it's richer is because now it encodes information of other tokens also so it
41:52
retains context so context Vector is an enriched
41:58
embedding Vector it combines information from all the other input
42:04
elements so in self in self attention context vectors play a very crucial role
42:11
their purpose is to create enrich representations of each element in an input sequence by incorporating
42:17
information from all the other elements in that sequence uh so this is again keep this
42:23
thing in mind that input embedding vectors only contain information about that that that word or that token it
42:31
might encode information about the meaning of that word and its position but it has no clue of the neighbors
42:37
context Vector has clue about the neighbors uh because neighbors are so important right when you look at a
42:43
sentence when you look at a paragraph individual tokens don't mean anything it's only how they relate with the
42:50
neighbors that essentially produces the context of that paragraph um and why is this needed in
42:56
llm it's needed to understand the relationship and relevance of words in a sentence to each other actually this is
43:04
that fundamental thing which has made llm so so much better so if you look at
43:09
this advancements in history right Elisa RNN alist El until your attention was
43:15
not there 2014 I believe is a very critical point that was 10 years back when the attention mechanism was
43:21
introduced and then people started thinking that oh instead of looking at words in isolation what if I take a step
43:27
back and try to see how different words essentially relate to each other so then we are exploiting the maximum richness
43:35
from text because just like images are made up of patterns of pixels text or
43:41
paragraphs only make sense if you take a look at all the words together and how
43:47
they relate to each other so now the question is that okay you have an input embedding Vector uh
43:54
let's say for next how do you convert it into a context Vector so you have an
44:00
input embedding Vector how do you go from the input embedding Vector to the context
44:05
vector and I want you to think about this from the first principles pause this video for a moment and think about
44:12
this right you have the input embedding vector and let's say you have these attention scores how will you modify the input
44:19
embedding Vector so that somehow these attention scores are taken into account
44:24
and you have a context vector so you can pause here for a moment first
44:32
you can try to also think about how these attention scores themselves are computed
44:38
um okay so the simplest thing to do is that let's say if you have
44:44
uh uh this Vector right and if you have all the other vectors why don't we take
44:49
a simple dot product so you have the input embedding Vector for next you have the input embedding Vector for the just
44:57
just take a DOT product between these two that will give you Alpha 2 one just take a DOT product between next and next
45:03
that will give you Alpha 22 uh then take a DOT product between next and day that
45:08
will give you Alpha 2 three then take a DOT product between next and is that will give you Alpha 24 and next take a
45:16
DOT product between next and bright that will give you Alpha 25 and once you have all these Alphas
45:22
you can simply do alpha 21 * X1 plus Alpha 2 2 * X2 + Alpha 2 3 *
45:31
X3 + Alpha 2 4 * X4 plus Alpha 25 *
45:40
X5 and why do we take a DOT product here essentially the reason you might think
45:46
of a DOT product is that a DOT product essentially encapsulates information about whether vectors are similar or
45:52
closer to each other or not right if you have one vector here V1 and if if you have another Vector here V2 the dot
45:59
product between them will be higher than let's say V1 and V3 so if two vectors are similar their
46:06
dot products will be higher and that's exactly what you wanted to quantify with the attention mechanism you might think
46:12
that I want to quantify whether two vectors are similar right so if next and the are more similar of course they
46:18
should have a higher attention score so this calculation seems to make sense to me I just calculate the dot products and
46:25
then I just scale the different vectors with their dot products and add it up so
46:31
whatever the sum would be that would be the context Vector now for next and similarly I can find context
46:38
vectors for all the other tokens as well what's wrong with this
46:44
approach why will this approach not work or why can't we just take simple dot
46:51
products to find the attention scores again you can pause here for a moment M and try to think of the context
46:59
which we are trying to encode um I want you to think from first
47:05
principles here I'm going to reveal the answer very soon all right so the main answer is
47:11
that let's say you consider this sentence the dog chased the ball but it couldn't catch it right the dog chased
47:18
the ball but it couldn't catch it and let's say I have the input embedding Vector for dog as this input
47:25
embedding Vector for ball as this and input embedding Vector for it as this okay and if it is my query Vector right
47:32
now um how did you decide to compute the attention score between a query vector
47:38
and other vectors you decided to take a DOT product right this is exactly what I'm going to do if it is my query Vector
47:44
to get the attention score between it and dog let me simply take a DOT product between it and dog and if I take a DOT
47:51
product it's 51 if I take a simple dot product between it and ball that is is
47:56
also 51 you see the problem here both the attention scores are completely
48:02
identical but that's not what I wanted when when you say but it couldn't catch
48:07
it it actually means the ball right the dog chased the ball but it couldn't catch it so the second it means the ball
48:16
not the dog so when I'm looking at it I need to pay more attention to ball not
48:21
the dog let let me write this with a
48:27
different ink so that this is clear I need to pay a lot more attention to ball
48:33
when I'm looking at it and not really the dog but that's not that's not what is
48:39
happening here if you take a simple dot product there is no provision for me to encode the information that ball should
48:46
be given more priority than dog both dog and ball should not be having the same
48:52
attention score when we say it this is actually this example is a
48:57
brilliant demonstration of why we should selectively attend to different tokens the dog chased the ball but it couldn't
49:03
catch it the first it is the first dog so if this was the query token it would
49:08
have attended more to dog but now this is the query token so it should attend more to ball I don't want both to have
49:16
the same attention score so simple dot product cannot distinguish between subtle contextual relationships here it
49:23
doesn't consider the context of Chase couldn't catch or linguistic Nuance such as the fact that catch is more likely to
49:30
refer to a moving object which is the ball so the main issue is that simple do
49:36
product only measures semantic similarity but it cannot deal with
49:41
contextual issues and many sentences might have contextual complexity like this right um and I need to encode a
49:49
mechanism so that I can capture these complexities and I don't know what that
49:55
mechanism would be so then we use the trick which researchers have used for a long period of time now if you don't
50:02
know what the underlying relationship between things is you just replace it with a neural network or a bunch of
50:08
trainable weight metries and let back propagation figure it out and that's
50:14
exactly what happened in the field of attention also researchers essentially could not figure out what that mechanism
50:21
can be so and that's where the field of machine learning deviates or deep learning Ates from physics right in
50:28
physics if you were stuck with this problem you would have spent 6 months one year trying to develop a law for the
50:34
underlying mechanism to capture complexities or underlying mechanism to capture the context but in the field of
50:41
deep learning you don't do that you say that I'll replace it with a bunch of matrices and I'll train these
50:48
matrices through back propagation so that's what researchers did right so they
50:54
invented new matrices which are let's say called as the query Matrix and the key Matrix what it means is that instead
51:01
of just looking at the input embedding representations What If I multiply every
51:07
input embedding with a matrix so if my query here is it right my query is it
51:15
I'll multiply it with something which is called as the query Matrix this can be a high dimensional
51:21
Matrix um for dog so dog and ball are the keys right
51:27
uh because keys are essentially if you have the query keys are essentially all the other tokens which you're looking
51:33
for so that's dog and ball so you you multiply both of them with a keys Matrix
51:38
now see the advantage here is that if a DOT product cannot get the contextual relationship you are hoping that these
51:45
WQ and WK you are not assuming these Matrix as anything you are just you will
51:50
initialize them randomly and then you will train them through back propagation
51:56
it's the same deep learning trick which researchers now have used for a very long time if you cannot figure out the
52:02
relationship yourself you take a step back and you let neural network do its job instead of restraining the neural
52:09
network by imposing some laws let it figure it out itself so you see the advantage is now we have multiple uh
52:16
trainable factors in our control so if WQ is let's say
52:21
3x3 and WK is 3x3 right um so dog ball
52:26
and it these are my keys and if these are the embeddings for these which we saw here also the input
52:34
embeddings which we saw and now I multiply these input embeddings with the query I multiply
52:41
this these I multiply these two with the keys so I will multiply so it will be
52:47
3x3 multiplied 3x 1 so this will be a 3x1 and this will also be a
52:55
3x1 so then the keys become .92 and 0.1 And1 1.8 And1 you see these values
53:03
changed because I multiplied them with the keys Matrix and the query is
53:09
it so it will be multiplied with the queries Matrix so that the query for it
53:15
will become 0.5.1 and. 5 1.0
53:21
And1 uh and so now if you plot this in Vector space this is the query Vector
53:27
this is the keys for ball and this is the keys for dog so now we have we are going from the input embedding space to
53:35
a different space which we get after multiplying with the queries and the key Matrix and now I will compute the
53:41
attention scores between these vectors not the original vectors so now if you compute the attention score between it
53:47
and the ball you'll see that it's 56 it and the it and the ball
53:53
is96 and if you compute the attention SC score between it and the dog that is 56
53:59
so here you see the attention score between it and ball is96 Which is higher
54:06
than the attention score between it and dog which is lower so these are clearly distinct
54:12
attention scores so adding these trainable matrices has actually helped
54:17
us why has it helped us because it has given a number of parameters to tune so
54:22
that we can encode some complex relationships between tokens so if you take a simple dot product the attention
54:29
scores will be identical but if you take if you have the query key Matrix we have
54:35
not yet seen the value Matrix we'll see that in the next class but essentially if you just have trainable
54:41
matrices then you can have attention scores which are different because now you suddenly have more parameters to
54:47
work with so if you got confused in this part let me repeat it once more um we started
54:54
this section by thinking that if you have an input embedding Vector right what can you do to the input embedding
55:00
Vector to get the context Vector so to get the context Vector we essentially need Alphas after you get the alphas
55:08
then you just have to multiply them with the um input embedding
55:13
vectors uh and then you will get the context Vector but then the question is that how do you get the alphas between
55:21
one uh input embedding vector and another input embedding Vector how do you get the attention scores the
55:27
simplest way is probably taking a DOT product but we saw that let's say if this is the sentence right and if it is
55:34
my query and if I I want to find the attention score between it this it dog
55:40
and ball I will take a DOT product between it and the ball first which comes out to be0 51 and I will take a
55:47
DOT product between it and the dog which comes out to be again 51 so the attention scores comes out to
55:54
be similar but this is not what I wanted because when I say it I want it to be
55:59
the ball so I want the attention score between it and ball to be much higher than the attention score between it and
56:07
dog so how to do it now dot product clearly does not have the complexity to
56:12
capture these contextual relationships I need more parameters to work with I need
56:18
some knobs which I don't know currently but let neural networks or let back propagation figure out what those knobs
56:24
can be at least let me initialize it randomly for now and that's where this new terminologies coming to the picture
56:31
right I want to have new trainable matrices let me call this query Matrix
56:37
and let me transform the input embeddings into another space by multiplying it with the query Matrix and
56:44
the input embeddings for the keys which are the dog and the ball they'll be multiplied with the keys Matrix and
56:51
we'll transform it into another space and then I will find the attention scores in that transformed vectors
56:56
between those transformed vectors and now if my model learns these parameters of these matrices correctly I can get
57:05
the model to learn that the attention score between it and the ball is96 Which
57:10
is higher than the attention score between it and the dog which is 56 don't worry about these
57:17
multiplications or mathematics right now I I'll do the mathematics in detail in the next lecture for now just remember
57:24
that we we don't know how to physically capture the contextual relationship so
57:30
it's like an easy way out it's a trick you you introduce the queries you introduce the keys randomly these are
57:36
random trainable matrices you initialize them randomly and then you train them so
57:42
you might have heard of this word query Keys there's actually no proper physical reason why they are introduced the only
57:49
reason they are introduced is because humans could not figure out how to capture these attention scores
57:54
themselves the only way we know is that okay if we cannot figure it out let me project my input embeddings into higher
58:02
Dimensions or different dimensions or let me have few trainable parameters to work with and then
58:08
hopefully the training itself will figure it out on its own and this trick humans have done in
58:14
the field of computer vision also if you train a CNN to distinguish between dogs and cats you cannot write down all the
58:22
features yourself you rely on a convolutional neural network to do that it's kind of a similar thing over here
58:30
in the next lecture we are actually going to see how do we exactly compute the queries Matrix the keys Matrix and
58:37
there is also one more Matrix which is called the values Matrix how that is used in the next token prediction task
58:45
that we are going to see in the next lecture so the next lecture is all about the next lecture is about the
58:52
mathematics of self attention mechanism what do we do with the queries Matrix key Matrix and the values Matrix exactly
59:00
how do we calculate the context vectors mathematically and from those context vectors
59:06
ultimately uh what do we do in all these steps to get the next token prediction
59:13
so next class is pretty much going to be a deep dive into this section which we
59:18
just saw uh and expanding it into a full lecture of mathematics but now in
59:23
today's lecture I just want to motivate this concept of queries keys and values values we have not seen yet we'll see
59:30
that in the next lecture all right everyone so this brings us to the end of today's lecture
59:36
which you can think of as a mixture of the history of the attention mechanism plus an introduction to self attention
59:43
for the next token prediction task as a summary remember how the attention mechanism has Evolved first we had uh
59:51
Elisa uh Elisa was a revolution at that time and it's pretty awesome considering it inv got invented in
59:58
1966 then came recurrent neural networks and lsms they had the context bottl niic
1:00:04
issue which means that all the context was compressed into just one hidden state to solve that we understood that
1:00:10
we needed to selectively pay attention to different parts of the input sequence and that is what is called as attention
1:00:17
so to encode that we introduced something called as the attention mechanism which computes the attention
1:00:22
scores between the decoded output or the decorder hidden States and the input hidden states that paper was the badana
1:00:30
attention mechanism published in 2014 uh that paper essentially still
1:00:35
had RNN so that was attention plus RNN in 2017 there came a paper in which
1:00:42
researchers realized that we don't even need rnns so they scrapped out rnns and
1:00:47
they came up with a new architecture called the Transformer architecture which had the attention mechanism at the
1:00:52
heart of it 2018 researchers modified the Transformer architecture they
1:00:57
scrapped the encoder kept the decoder and had this architecture in which the
1:01:03
attention mechanism was again at the heart of it uh so this until now the attention
1:01:09
mechanism was from one sequence to another sequence then when we talk about self attention we essentially look at
1:01:16
just one sequence because that will be used for next token prediction tasks so
1:01:22
in next token prediction tasks like GPT we use self attention where we look at one token and how it
1:01:28
attends to its surrounding or neighboring tokens so the token which we are looking
1:01:33
at is called as query and the other tokens are called as keys and we want to find the attention score between the
1:01:39
query vector and the keys we realize that the main purpose of the attention mechanism is to get these attention
1:01:46
scores and to convert it into context Vector context Vector is a more enriched
1:01:51
version of the input embedding Vector because it also contains information about how one token relates to its
1:01:58
neighbors to get these attention scores the naive way or the simplest way to think about it is just to take a DOT
1:02:05
product between vectors but we realize that that's not the best way to go about it because just taking a simple dot
1:02:12
product can't capture subtle contextual relationships like we saw in this
1:02:17
example the dog chased the ball but it couldn't catch it the first it is the dog the second it is the ball to capture
1:02:24
such contextual complexities we need to add trainable weight matrices so we need to increase
1:02:32
the parameters so that we have different knobs to play around with these trainable matrices are called as the
1:02:38
query weight Matrix and the key weight Matrix there is also value weight Matrix which we'll see in the next class the
1:02:44
input embedding for the all input embeddings are multiplied with the query weight Matrix to get the query Matrix
1:02:51
and also we have the keys Matrix like that so then the attention scores are not found between the input embeddings
1:02:57
of the vector they are found between the queries and the keys and since we have a flexibility of
1:03:04
so many parameters to play with we hope that when we train the parameters they will learn that the attention score
1:03:10
between the it the second it second it and the ball is higher than
1:03:18
the attention score between the second it and the dog so it captures more
1:03:24
contextual complexities so addition of these trainable weight mates captures more contextual complexities and that's
1:03:32
why we humans added these weight matrices and then we call them queries
1:03:37
keys and values because it it sounds cool and it also it relates to the field of information Theory but if you look at
1:03:44
it deeply we cannot figure out the rule for this attention mechanism ourselves
1:03:50
do product fails so we cannot figure out how to get these attention scores how to compute them ourselves so we turn to
1:03:56
neural networks to do the job for us um all right so thanks a lot everyone
1:04:03
in the next lecture we'll be diving deep into the mathematics behind self attention in the lecture after that
1:04:09
we'll look at multi-head attention and only then we'll be so this is the multihead attention notes and only then
1:04:16
we'll be truly ready to understand uh the key value
1:04:22
cache so let me see where that is yeah only then will be truly ready to
1:04:27
understand the key value cach which serves as the segue for the multi-head latent attention which is
1:04:33
MLA this series is going to be a bit deep but I'm trying to make the lectures as long as possible so that I don't miss
1:04:41
out anything this is for serious Learners so please make notes as you are watching this series and it will be
1:04:47
incredibly useful for you thanks a lot everyone and I look forward to seeing you in the next lecture






