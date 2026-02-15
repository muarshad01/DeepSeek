#### [invideo - Create Videos Without Limits](https://invideo.io/?utm_source=google&utm_medium=cpc&utm_campaign=Top16_Search_Brand_Exact_EN&adset_name=Invideo_AI&keyword=invideo.%20io&network=g&device=c&utm_term=invideo.%20io&utm_content=Invideo_AI&matchtype=e&placement=g&campaign_id=18035330768&adset_id=152533182854&ad_id=674819400171&gad_source=1&gad_campaignid=18035330768&gbraid=0AAAAACqfi_AodnB9BAf3Yt3ThUh5Qgi6U&gclid=EAIaIQobChMIqujNluXbkgMV3UpHAR2hexKwEAAYASAAEgKn4fD_BwE)
* __Prompt__: Create a hyper realistic video of a commertical of a premium luxury watch. Make it cinematic, use closeup of the watch and its parts. Use American female voice for english narration.

***

#### Scaled Dot-Product Attention

$$\text{Attention}(Q,K,V)=\text{softmax}\bigg(\frac{QK^T}{\sqrt{d_{k}}}\bigg)V$$


#### Softmax

$$\\{x_1, x_2, x_3, x_4, x_5, x_6\\}$$

$$\big\{\frac{e^{x_1}}{\text{sum}},\frac{e^{x_2}}{\text{sum}},\frac{e^{x_3}}{\text{sum}},\frac{e^{x_4}}{\text{sum}},\frac{e^{x_5}}{\text{sum}},\frac{e^{x_6}}{\text{sum}},\big\}$$

$$\text{sum} = e^{x_2} + e^{x_2} + e^{x_3} + e^{x_4} + e^{x_5} + e^{x_6}$$

***

* 5:00

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

chart I can see how much attention needs to be paid to each token whereas if you see these values right now um let's
25:12
see um yeah if you see these values right now for the second row that is the
25:17
attention score for next you'll see that these values don't really sum up to one
25:22
which means that I cannot make statements like give 10% attention to the first token 18 % attention to Second
25:29
token it will not work like that the rows the values in the rows here do not sum up to one and that is the main
25:36
problem so the next step which is Step number three is to make sure that the attention scores are converted into
25:43
something which is called as attention weights and to go from attention scores
25:49
to attention weights we are going to apply the soft Max operation soft Max essentially means that let's say we take
25:55
a look at this row um
26:00
and let me write it down in column format right now so then in column format this will become .1
26:07
1.8 6.1 and .1 what I want to do is I want to somehow convert all of these
26:14
values so that they lie between 0o to one and they also sum up to one so the
26:19
softmax operation essentially what it does is that if this is X1 X2 X3 X4 and X5 soft Max what it does it
26:30
replaces X1 with e to X1 divided by summation it replaces X2 with e to X2
26:36
divided by summation it replaces X3 with e to X3 ided summation e to X4 ided
26:43
summation and E to X5 / summation uh now what exactly is
26:51
summation summation is just e to X1 + e to X2 + e to X3 plus e to X4 + e to
27:01
X5 now if you take a look at all of these five values you'll see that if you
27:06
add them the numerator will be e to X1 plus e to X2 plus e to X3 plus e to X4
27:11
plus e to X5 that is equal to summation which is the denominator so all of these
27:17
will definitely add up to one and they will lie between 0 to one soft Max also
27:22
has the additional important property that it gives a lot of weightage to very high values and very low weightage to
27:29
very low values this makes the classification very easy um so essentially that's the
27:36
softmax operation which we are going to implement but the main problem here is that I told you right softmax gives very
27:43
high attention to values which are very high and it does not pay that much attention to values which are very low
27:50
and that's a big problem for us so let's say if the attention let's say if this if the atten
27:58
ion scores are something like this
28:04
okay uh now you see this value is very high right if you apply softmax to this
28:10
the way softmax will work is that it will put .95 or something to this value
28:15
and make sure that all the other values are actually very low and if these are our attention
28:23
weights that's not very good for us because then we'll pay a lot of attention to one key and will not pay
28:30
attention to all the other keys at all so that's why there is a scaling which needs to be performed before we
28:36
apply the soft Max before we apply the soft Max we need to make sure that all of these values are divided by some
28:43
value and only then we will apply the soft Max so now let me talk about that
28:48
part a bit um let's go to this
28:56
part yeah so here I want to first of all explain um the issues with softmax right
29:03
so what I'm doing is that let's say these are my um
29:09
attention score values and then I apply softmax to these values right U so we can see that
29:17
softmax for all of these values is almost equally equally similar in terms of range that's good for us but now what
29:24
I want to do is that I want to multiply all of these values with eight and then let us apply soft Max so if all of these
29:31
values are multiplied by 8 which means that some values will be very large and some values will not be that will not be
29:38
that high right so if if I multiply all of these values by eight and then I apply softmax you will see
29:45
that softmax places a lot of importance on this value which is 08 and it gives
29:50
negligible importance to some other values see that is what happens when the
29:56
values are very large before applying soft Max and that is what I've have mentioned
30:02
over here the soft Max function is sensitive to the magnitude of its inputs
30:07
when the inputs are very large the differences between exponential values of each input become much much more
30:13
pronounced and this causes softmax output to become peaky what peaky output means is that it gives a lot of
30:19
importance to some values and very low importance to others so in attention
30:25
mechanism this is not very good because because it if it's a sharp soft Max distribution then the model becomes very
30:32
confident in one particular key so now you see the model becomes very confident
30:38
in this key and it will give very low confidence to the other keys that leads
30:43
to very unstable training when we look at the Transformer architecture later and we do not want that so that is why
30:50
we need to scale um that is why we need to scale this
30:56
this this vector before we apply soft Max so the value with which this Vector
31:02
is actually scaled is square root of the keys Dimension right so the keys Dimension here is uh in this case it's
31:10
5x4 right so the output Dimension is equal to 5 so it scaled by square the
31:16
output Dimension is equal to four sorry so it scaled by square root of 4 and now you must be thinking that why
31:23
is it scaled by square root of 4 why just why don't we just scale it with the key Dimension or Dimension raised to two
31:31
or something like that why only square root of the keys Dimension what's so special about the square
31:37
root so the main idea here is that uh the reason we scale with the square root
31:44
of the keys Dimension is because of a concept of variance so if you um if you
31:51
take if let's say this is my queries this is my queries Vector which is a six dimensional vector
31:58
and this is my keys transpose which is a six dimensional Vector this is my keys transpose okay
32:06
and if I multiply these vectors I'll get some values right but remember in the multiplication what is happening is that
32:12
each value is getting multiplied and then summed
32:17
up now what usually happens is that
32:23
if there are two random vectors right if there are two random ROM vectors whose
32:29
dimensions are six and if I sample 100 such random vectors so I'm going to do
32:36
something like this I I take 100 vectors of the
32:43
queries and I take 100 vectors of the keys
32:48
transpose and then what I do is that I collect the values of queries multiplied by key keys transpose for all of these
32:55
100 what happens is that the variance of
33:01
this product so now I can take these 100 values and I can compute their variance
33:06
right how the distribution is the variance of this product actually
33:14
scales with the keys Dimension so as has been mentioned here um the variance of
33:20
this product scales by the square root of the keys Dimension actually um scales with square root of
33:28
the keys Dimension which means that as the keys Dimensions goes on increasing a
33:33
lot or rather as the dimensions of keys transpose here let's say if that increases the variance of this product
33:40
increases a lot which means that since the queries and the keys transpose are initialized randomly these are random
33:46
vectors right their product can be usely varying from some very high values to
33:51
some very low values and we want to avoid that as much as possible we want to make sure that this product variance
33:58
is equal to one so that the product does not Wily oscillate the queries and the
34:03
keys are defined randomly at the start right and the dimensions can be very high so if the dimension is very high
34:11
and if the variance actually scales with the square root of this Dimension then that's not very good for us because it
34:17
will also make the learning unstable and to illustrate this concept further I want to explain it to you with the case
34:25
of dice right so if you're rolling one dice let's say um and it just has one to
34:30
six numbers right the average of this is let's say 3.5 and the variance is relatively small that is let's say 2.9
34:38
there are predictable outcomes but now what I do is that I roll uh I roll the
34:44
dice and then I sum the output of 100 I roll and sum 100 dice basically so if
34:51
you roll a dice 100 times and you'll sum the output what happens here is that if
34:56
summation is involved the mean is around 350 but the variance grow significantly to around
35:03
290 so the output becomes unpredictable and the dot product
35:09
without normalization so if we if we just take the queries multiplied by the keys transpose and if we do not divide
35:15
it by the square root of the keys Dimension we'll see that increasing the
35:21
number of Dimensions is like rolling more Dice and summing the results why because dot product essentially is where
35:28
we sum up the product right where we sum up if the dimension is 100 we sum up 100 entries of the queries multiplied by
35:34
Keys transpose so increasing the number of Dimensions is like rolling more Dice and
35:40
summing the number of results each Dimension contributes some variance and as the dimension grows variance
35:47
accumulates um and so here I'm trying to say that as the query and the keys
35:53
Dimension grows the variance increases a lot and the dot products before soft Max
35:58
become either very large or it becomes very small because the variance is large
36:04
and that makes the attention weights unstable and so the training procedure also becomes unstable when we divide by
36:10
square root of D it scales down the variance of this do product and brings it equal to
36:15
one and that stabilizes the expected outcomes the attention weights become more stable and they become more
36:21
predictable I have actually explained this using the code below so here what we'll do is that we'll take thousand
36:29
trials okay and in each trial we'll generate a so first we'll do a five
36:34
dimensional query and the key vector and then we'll do 100 dimensional query and
36:40
the key Vector right so let's say the dimension is equal to five we'll generate queries and key vectors of
36:46
Dimension five and we'll do that thousand times we'll compute the dot product
36:51
between the query and the key for each trial um and then in one case we'll
36:56
divide by the square root of the dimension and in the other case we'll just take the dot product like that
37:03
without without doing the scaling and we'll collect all the results of the Thousand trials and then we'll find the
37:09
variance we'll find the variance before scaling and we'll find the variance after scaling so remember we are taking
37:15
th000 query vectors th000 key vectors we are taking the dot product and in one case we are scaling it in one case we
37:21
are not scaling it and then we are finding the variance before scaling and after scaling so if you run this you see
37:27
that for the dimension of five variance before scaling is equal to five for the
37:34
dimension of 100 variance before scaling is actually 100 so the variance actually
37:39
directly grows with Dimension uh the variance is directly proportional to the dimension if you see
37:45
but after scaling the good thing is which happens is after you scale with
37:51
the square root of uh the dimension the variance almost becomes equal to one
37:57
that that's very cool right so now here it's clearly proved that if you divide the product if you divide the dot
38:03
product of the queries and the keys with the square root of the keys Dimension
38:09
then uh what you ultimately get is that um after scaling whether the dimension
38:16
is five or whether the dimension is 100 the variance of the product is constrained this means that the queries
38:22
multipli multiplied by Keys transpose the values won't blow up to very high values or very low values the variance
38:28
will remain equal to one that will lead to very stable training procedure so that's why what we do is
38:36
that in Step number three the attention scores are scaled by square root of the
38:42
keyys dimension which means that we take the attention scores and we divide by square root of Key's Dimension Key's
38:48
Dimension essentially is every key Vector what's the dimension of it in this case it's equal to four right so we
38:54
divide by square root of 4 and then we apply soft Max so after this division is done uh after this
39:02
division is done by square root of uh the keys Dimension we apply soft Max
39:07
essentially we apply this function and when we apply the soft Max attention
39:12
scores are converted into what is called as attention weights so that's the key difference here between attention scores
39:19
and attention weights attention scores are not normalized and attention weights
39:25
are normalized so if you look at every row over here you will see that every row essentially sums up to one and now
39:32
we can make the quantitative statements or the qualitative statements which I was mentioning right this second row
39:37
corresponds to next so now I can say that we can pay 10% attention between
39:43
next and the we can pay 50% attention between next and next we can pay 20%
39:49
attention to next and day we can pay 20% attention to next and is and we can pay
39:55
10 sorry 10% attention to next is and we can pay 10% attention to next and bright
40:00
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




















