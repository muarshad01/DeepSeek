#### [Invideo AI - Video Generator | No Editing Skills Needed](https://invideo.io/?utm_source=google&utm_medium=cpc&utm_campaign=Top16_Search_Brand_Exact_EN&adset_name=InVideo&keyword=invideo&network=g&device=c&utm_term=invideo&utm_content=InVideo&matchtype=e&placement=g&campaign_id=18035330768&adset_id=140632017072&ad_id=616240030555&gad_source=1&gad_campaignid=18035330768&gbraid=0AAAAACqfi_CasH_dati6efWraWC4sWV3x&gclid=Cj0KCQiA7rDMBhCjARIsAGDBuECUPsCNxYndLvG9dbQcl3duZVnCdxZL4bu-YboI7x4VTCTYt6KBjfcaAvMFEALw_wcB)

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

#### Architecture of LLM

| Model | Parameters |
|---|---|
| GPT-3   | 175 Billion|
| GPT-4   | 1 Trillion |
| GPT-4.5 | 5-10 Trillion |

***

* [ChatGPT](https://chatgpt.com/)

#### Transformer Block
1. Layer Norm 1
2. Multi-head Attion --> MLA
3. Dropout
4. Layer Norm 2
5. Feed Forward NN --> MoE
6. Dropout

***

```
A true friend accepts you
```

* __Phase-1__: Isolation
  * The word is isolated from its neighbors
* __Phase-2__: Token ID assignment
  * Book of Token IDs (Vocabulary)
    * Words
    * Sub-words
    * Characters
  * Byte Pair Encoding (BPE) 
* __Phase-3__: Token embedding assignment

***

* __Phase-4__: Positional embedding assignment (Your position among neighbors matter!)

```
The dog chased another dog
```

* __Phase-5__: Add token embedding to positional embedding.
  * Input embedding = Token embedding + Positional embedding

* __Phase-6__: Now, you're finally ready to onboard the train to the Transformer block.


***

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


***

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



***


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



***


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

***
