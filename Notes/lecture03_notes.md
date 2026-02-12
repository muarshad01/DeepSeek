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



