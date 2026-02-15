#### [invideo - Create Videos Without Limits](https://invideo.io/?utm_source=google&utm_medium=cpc&utm_campaign=Top16_Search_Brand_Exact_EN&adset_name=Invideo_AI&keyword=invideo.%20io&network=g&device=c&utm_term=invideo.%20io&utm_content=Invideo_AI&matchtype=e&placement=g&campaign_id=18035330768&adset_id=152533182854&ad_id=674819400171&gad_source=1&gad_campaignid=18035330768&gbraid=0AAAAACqfi_AodnB9BAf3Yt3ThUh5Qgi6U&gclid=EAIaIQobChMIqujNluXbkgMV3UpHAR2hexKwEAAYASAAEgKn4fD_BwE)
* __Prompt__: Create a hyper realistic video of a commertical of a premium luxury watch. Make it cinematic, use closeup of the watch and its parts. Use American female voice for english narration.

***

#### Scaled Dot-Product Attention

$$\text{Attention}(Q,K,V)=\text{softmax}\bigg(\frac{QK^T}{\sqrt{d_{k}}}\bigg)V$$

***

#### Self Attention with Trainable Weights

```
The next day is bright
```

* Input Embedding = Token Embedding + Positional Embedding

####  Example
* Words = 5
* d_input = Input Emdebbing dimenesion = 8
* (5 X 8)

#### Trainable Weight Metrices
* Query  = W_q = (8,4) = (d_input, d_output)
* Keys   = W_k = (8,4)
* Values = W_v = (8,4)

* For W_q, W_k, and W_v usually d_input = d_output

* We want to tranform input embeddings into different space, so that, our expressivity increses and we can capture undering complexitites which cann't be done through a simple dot product. 

***

* 20:00

* $$\text{Attention ~Score} = Q \times K^{T}$$

***

* 25:00

#### Softmax

$$\\{x_1, x_2, x_3, x_4, x_5, x_6\\}$$

$$\bigg\\{\frac{e^{x_1}}{\text{sum}},\frac{e^{x_2}}{\text{sum}},\frac{e^{x_3}}{\text{sum}},\frac{e^{x_4}}{\text{sum}},\frac{e^{x_5}}{\text{sum}},\frac{e^{x_6}}{\text{sum}},\bigg\\}$$

$$\text{sum} = e^{x_1} + e^{x_2} + e^{x_3} + e^{x_4} + e^{x_5} + e^{x_6}$$


* __Peaky Output__: Softmax gives more attention to higher values and less attention to lower values.
* __Unstable Training__:

#### Scaling
* Why $\sqrt{d_{keys}}$
* Variance scales as $\sqrt{d_{keys}}$

***

* 35:00

* Attention Score
* Attentin Weight (are normalized)

***

* 40:00

***

* 45:00

***

through this self attention class that's it is it's as simple as that to retain the dimensions of what we
50:25
had in the learn code what we can actually do is that we can make sure the input is eight Dimensions so uh let's
50:33
let's take it let's make the code similar to what we had here so the next day is bright right let's say this is
50:41
the next day
50:48
is the next day is bright and let's say so now this is not
50:55
needed
51:04
okay and now we in the code or in the Whiteboard we have seen that there is eight dimensions for every
51:11
every Vector so let me just copy paste it to make eight dimensions and then add
51:17
some random values here um and let me just add this the the same thing here for the sake of
51:24
Simplicity now so I'm making everything 8 Dimensions here so that it matches what we had on the Whiteboard okay and D
51:31
in is now equal to 8 but my D out which I taken over here was equal to four so
51:37
I'll just take D out equal to 4 and then let me run this block so here you see it runs almost immediately and then I have
51:44
the context Vector Matrix let's check the dimensions the dimensions here of the context Vector Matrix is that we
51:51
have five rows essentially and four columns this is exactly what we had seen over here the context vector or the
51:58
context weight context Vector Matrix is five rows and four columns that's
52:03
exactly what we have implemented in the code right now in the next class what we'll see is that we'll Implement
52:09
something which is called as causal attention in the causal attention we will hide out the future attention
52:15
weights which are not uh which are not needed or which are not available to the current token so that uh we only look at
52:23
the past tokens before predicting the next value so that's called causal attention we'll see that in the next
52:29
lecture and then we'll move to multi-head attention I know that these lectures are a bit long and uh I'm
52:35
repeating some elements with respect to attention Transformer blocks causal attention context vectors Etc but I
52:42
believe it's extremely crucial for us to understand this if we don't understand this the learning of multi-ad latent
52:50
attention will not be very strong I want all of us to be on the same page when we start the multi-head latent attention
52:56
part so all of you really need to understand the nuts and bols of self attention and multi-head attention and
53:03
that's why I'm making this lectures in so much more detail I'll share the code files with you and uh I hope that as you
53:11
are going through these lectures you also make notes so write this down on a piece of paper make sure that you
53:17
familiarize yourself with the calculations later we'll go to multi-head attention and later we go to
53:23
multi-ad latent attention so remember that it's a m step Journey it's not easy this not going to be a 30 minute course
53:30
it's going to be a course of 35 to 40 videos and it will be a very very detailed course I intend to make it like
53:37
a university lecture right so I could have directly started with latent attention but that would be very
53:44
difficult I think to directly understand and uh I want to make these lectures as
53:49
useful for an audience which is seeing this series for the first time also so I
53:54
hope you are enjoying this series and I look forward to seeing you in the next lecture thank you










































