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

* __Phase-7__: Different compartments of a Transformer block
1. Layer Norm 1
2. Multi-head Attion --> MLA
3. Dropout (Improves generalization performance; prevents over-fitting)
4. Skip connection or shortcut connnection (help gradient to flow through an alternate path; vanishing gradient problem)
5. Layer Norm 2
6. Feed Forward NN (Expansion / Contraction) --> MoE
7. Dropout

***

* __Phase-9__: Normalization layer

* __Phase-10__:

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


