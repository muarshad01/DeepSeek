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


so that is how I can make qualitative statements now because all the elements of each row essentially sum up to
40:07
one that's the difference between attention weights and attention scores attention weights are normalized they
40:13
sum up to one whereas attention scores are not normalized and now what we'll do is that
40:19
we'll come to step number four in this step we actually compute the context vectors from the attention weights let's
40:27
see how that is done so until now we have the attention weights that's again a 5x5 Matrix and
40:34
when you see this Matrix again try to visualize what every row represents so the second row here represents if next
40:40
is the query how much attention should I give to the next day is bright and all of these values will now sum up to one
40:47
because they are normalized in the last step what we do is that remember until now we have not used the values vectors
40:53
at all in the last step these values vectors come in into the picture where
40:59


***


uh what we do is that the attention weight Matrix which is a 5x5 Matrix it simply multiplied with the values Vector
41:06
so the values Vector is 5x4 right and that gives us the context Vector the context Vector has five rows
41:13
and it has four columns the first row corresponds to the second row corresponds to
41:19
next U third row corresponds to day is and bright the next day is
41:26
bright and and now you can see that initially we started with input embedding vectors for the next day is
41:32
bright and after all these Steps step number one step number two step number three and step number four we have got
41:38
the context vectors for each of these tokens now now remember these context vectors are much more richer than the
41:45
input embedding Vector predominantly because their calculation involves the attention
41:51
weights uh so I just shown this calculation here mathematically the attention weights it's a 5x5 Matrix
41:57
which has been shown over here the values is a 5x4 matrix and so the context Vector Matrix is a 5x4 matrix so
42:05
for example the second row corresponds to the context Vector of next the last row over here corresponds
42:12
to the context Vector for bright so every context Vector is a four dimensional
42:18
Vector so here I just want to have a small section on the intuition behind the context Vector calculation so the
42:25
way the context Vector is actually Cal relateded is as follows right let's say if we want to find the context Vector
42:31
for next we have found the attention weights right between next and all the
42:36
other tokens so we have found that when you have next we should pay 10% attention to the we should pay 50%
42:44
attention to next we should pay 20% attention to day we should pay 10%
42:50
attention to is and we should pay 10% attention to Bright we have understood
42:56
this based on these attention weight values uh sorry these are tension weights and we have the value Vector so
43:02
you can think of the value Vector essentially as an input embedding Vector itself but it's transformed to a higher
43:08
Dimension or different dimensional space so now if we want to find the context
43:13
Vector for next what we essentially do is that uh we know that next gives 10%
43:20
attention to the so the so this is the uh Vector values vector for the so we'll
43:28
scale this Vector with 10% or we'll multiply this with 0.1 then we know that uh 0.5 which is
43:37
50% attention to next so I have written these attention weights here 0.1.5 21.1
43:44
so we give 0.1 importance to the first row which is the we give 0. five importance to the second row which is
43:50
next we give 20% importance to the third row which is day so we multiply the
43:56
third row with point 2 we multiply the fourth row with 0.1 and we multiply the
44:02
fifth row with 0.1 and all of these multiplications are then added together so we scale the
44:10
first Vector with 01 we scale the second Vector with 0. 5 we scale the third
44:15
Vector with Point 2 we scale the fourth Vector with point1 and we scale the fifth Vector with 01 and then add all
44:21
the scaled vectors together this is how we get the ultimate context vector
44:26
essenti after all this addition that's how we get the context Vector for next and to
44:33
represent it visually it looks something like this the next day is bright if you see the blue vectors they are the input
44:39
embedding vectors to get the context Vector we scale each vector by the amount of importance it needs to be paid
44:45
so the right that's 10% importance so we scale it with 0.1 next is 50% importance
44:52
so we scale it with 0.5 day is 20% importance so we scale it with 2 is is
44:58
10% importance so we scale it with 01 and bright is 10% importance we scale it with 0.1 all of the scaled vectors are
45:05
shown by this green and then all these green vectors are added together and that gives me the context Vector for
45:12
next so in this one figure you can see how the context Vector differs from the input embedding Vector right the input
45:19
embedding Vector for next only contains information about the meaning of that token but the context Vector for next
45:25
now contains information of all these attention weights also and all this
45:31
weight weighted some um ultimately gives the context Vector so this is how the
45:37
attention scores are added together to get the context Vector itself and this
45:43
is how the context Vector differs from the input embedding Vector now through this same visual you can try to see how
45:49
to get the context Vector for all the other tokens also if you have to get the context Vector for bright you just take
45:56
a look at the attention scores you multiply the first row with 0.1 you multiply the second row with 0.05 you
46:03
multiply the third row with 0.1 you multiply the fourth row with 0 25 and
46:08
you multiply the fifth row with 0.5 and then you add all these together so that will give you the context Vector for
46:15
bright which is essentially this this last row over here that's how you get the context Vector from the attention
46:21
weight Matrix and the value Matrix and to understand the visualization behind the context vector calculation look at
46:28
this diagram in this one diagram you can see how the attention weights are used as scaling factors and then the weighted
46:35
sum is essentially taken to convert US to take us from the input embedding space to the context Vector
46:41
space so this is the step number four which is essentially uh getting the context Vector Matrix and then that's it
46:48
in Step number five I have just sumarize this below we have the input embedding Matrix um we have the input embedding
46:56
Matrix and then we have the self attention layer so where whenever you see the self attention layer right now
47:02
it means all of these steps which have been implemented over here step number one is multiplication with the WQ WK and
47:08
WV to get the query vectors the keys vectors and value vectors step number two is multiplying the queries with the
47:15
keys transpose to get the attention scores step number three is scaling the
47:20
attention scores with square root of the keys Dimension and applying the soft Max to get the attention weights step number
47:26
number four is multiplying the attention weights with the values Matrix to get the context Vector Matrix that's it all
47:33
these four steps essentially are involved in the self attention module which ultimately takes the input
47:38
embedding Matrix and converts it into a context Vector Matrix that's it that's
47:44
the whole process which is going on in the attention mechanism or in the self attention mechanism and this process
47:50
Powers the Transformer block which is at the core of while language models work so well so
47:57
now if you take a look at our lecture where we had seen the different components of the Transformer block this
48:04
multi-head attention is where all the magic happens right we have the input embedding vectors as an input and the
48:09
context vectors as output and now you know how the context vectors are calculated from the input embedding
48:17
vectors why this is why is this called multihead attention we'll see that in the next lecture but for now I hope you
48:24
have understood the mechanism behind the self attention uh behind self
48:29
attention and I have just written a small code over here to demonstrate this to you so let's call this a self
48:36
attention class so we initialize the query query weight Matrix the key weight Matrix and the value we Matrix initially
48:43
we initialize them randomly and we set bias equal to false because we just have to multiply the input embedding Matrix
48:50
with these um and then in the forward pass what we do is we get the keys vectors the query vectors and the value
48:57
vectors by multiplying the input embedding Matrix with the trainable key Matrix trainable query Matrix and
49:03
trainable value Matrix that's essentially uh step number one which we looked
49:09
at then we come to step number two where we get the attention scores by multiplying the queries with the keys
49:15
transpose right then we come to step number three in Step number three we
49:20
essentially first divide by the square root of keys Dimension and then we apply soft Max and then then finally we come
49:27
to step number four which is multiplying the attention weights with the value Matrix to get the context Vector Matrix
49:34
that's it so I could have directly explained to you this code because it just 10 to 11 lines of code but to
49:41
understand these Matrix multiplications it's very important for us to write it down on a whiteboard so that we can
49:46
visualize the dimensions so this is how the context uh Vector weight Matrix is calculated you
49:52
can run this block I'll share this code also and then you can take it any input so here I'm taking an input your journey
49:59
starts with one step and the input DN is equal to 3 over here the input embedding
50:05
Dimension and I'm assuming the output embedding Dimension equal to two and let me just run this so here we can see that
50:11
the input embedding D input embedding Matrix is ultimately converted into the context Vector Matrix when we pass it
50:18
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








































