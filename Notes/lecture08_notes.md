#### Example
* __Step-1__: Start with 1 input batch
* X: (1, 3, 6)
* (b, num_tokens, d_in) = (1, 3, 6)
  * batch = 1
  * num_tokens = 3
  * d_in = 6

```python
b, num_tokens, d_in = x.shape
```

***

* 10:00
  
* __Step-2__: Decide (d_out, num_heads) = (6,2)

$$\text{head-dim} = \frac{d_{out}}{num_{heads}} = \frac{6}{2} = 3$$

* __Step-3__: Initialize trainable weight matrices for Key, query, value (W_k, W_q, W_v) 
  * W_k (d_in, d_out) = (6,6)
  * W_q (d_in, d_out) = (6,6) 
  * W_v (d_in, d_out) = (6,6) 

* __Step-4__: Calculate Keys, Queries, Value Matrix (Input X W_k, Input X W_q, Input X W_v)
  * Keyes (b, num_tokens, d_out) = (1 X 3 X 6) 
  * Queries (b, num_tokens, d_out) = (1 X 3 X 6) 
  * Values (b, num_tokens, d_out) = (1 X 3 X 6) 

***

* 15:00

* __Step-5__: Unroll last dimension of Keys, Queries, and Values to include num_heads and head_dim
* (b, num_tokesn, d_out) = (b, num_tokesn, head_dim, num_heads)
$$\text{head-dim} = \frac{d_{out}}{num_{heads}} = \frac{6}{2} = 3$$

***

* 20:00


columns right number of tokens comma D out which was this is just three rows
20:32
and six columns but now 3x 2 comma 3 you can visualize it like this there are three so this is um this is token
20:42
one this is token one this is token two and this is token three right and
20:49
earlier each token essentially had six colums associated with it the first
20:54
token had these six columns associated with it because the output Dimension was six but now you see these six columns
21:03
these six columns have been unrolled into a three three three column and
21:08
three column so essentially what is done is that uh after you divide it into two
21:14
parts um this thing is essentially brought over
21:22
here this thing is essentially brought over here and this is essentially
21:28
brought over here so now what I have is that 1 2 3 and let me write it by Brown
21:37
1 2 3 then um 1 2 3 and let me show this also by
21:47
Brown 1 2 3 and uh
21:52
last I have 1 2 3 and and brown
22:00
again 1 2 3 so what is done is that my
22:05
token one this is now my token one this is now my token
22:13
two and this is now my token number three and token one instead of token one
22:19
having all these six values together in a row the first row now corresponds to my head
22:25
one which are these three values over here and and the second row now corresponds to head number two which are
22:32
these three values over here similarly for token two this is my head
22:37
one and this is my head two similarly for token number three
22:43
this is my head one and this is my head number
22:49
two so this is what it means when we say convert a 1x 3A 6 or 3x 6 to 3x 2A 3 so
22:57
now this is a 3x2 3 so this is exactly what I have written over here 1X 3A 2A 3 looks like this
23:05
there are three why is it 3 comma 2 comma 3 because there are three tokens three tokens each token has two rows and
23:12
three columns so token number one has uh sorry each token has two rows and three
23:18
columns so token number one has two rows and three columns token number two has two rows and three columns token number
23:24
three has two rows and three columns this is exactly what can be seen over over here token number one uh token
23:31
number one has two rows and three columns token number two has two rows and three columns token number three has
23:37
two rows and three columns and what does each row in each token correspond to the first row corresponds to head number one
23:44
and the second row corresponds to head number two so now imagine that one token right um we are looking at the queries
23:52
so the first token had some sort of a input embedding that is split into two parts half of it goes to head number one
23:59
which is the first row and half of it goes to head number two which is the second row and that's done in a similar
24:06
way for token number two and token number three so this is the reshaped queries Matrix so essentially reshaping
24:14
just means splitting it into two so visually splitting the Matrix into two looks easier right but when in the code
24:21
you see these unrolling parts in the code when you see these unrolling parts it just gets very difficult to visualize
24:28
but here I'm deliberately showing this visualization to you so that it's actually very easy when you what does it
24:34
mean to unroll the last Dimension to unroll the last Dimension just means that you have this full Dimension which
24:40
is six dimensional token you split it into two and bring the bring the second half below the first part so that leads
24:47
to a 1x 3 by 1A 3A 2A 3 reshaped queries Matrix a 1A 3A 2A 3 reshaped Keys Matrix
24:56
and a 1x3 1 3A 2A 3 reshaped values
25:02
Matrix so this is done in the code also in the code what we have written here is
25:07
that unroll the last Dimension so earlier we had B comma number of tokens comma D out and now this is changed to B
25:15
comma number of tokens comma number of heads comma head Dimension number of heads is equal to two and head
25:22
Dimensions is three so that's 1A 3A 2A 3 that's exactly what what's written here
25:27
we are going to unroll this to B comma number of tokens comma number of heads comma head Dimension and the way this is
25:34
done is that the keys Matrix which was originally there right we do keys. view B comma number of tokens comma number of
25:41
heads comma head Dimension similarly for values we do values. view B comma number
25:47
of tokens comma number of heads comma head Dimension similarly for the queries
25:52
we do queries. view B comma number of tokens comma number of heads comma head
25:57
dimension so these are the keys values and queries which are reshaped so until now we are at this
26:05
part where we have reshaped the keys saries and values to take into account multiple attention heads then what we
26:11
are going to do is that uh so the keys queries and the
26:17
values have been obtained then we what we are going to do we are going to group The matrices by the number of heads so
26:23
you see the problem here is that we have grouped by the number of tokens right we see the token token one and within the
26:29
token one there is the head one and head two but now what we have to do is that we have to group it by the heads so
26:36
instead of 1A 3 comma 2 comma 3 I want 1 comma 2 comma 3 comma 3 which means I
26:42
want to Interchange this and this so essentially I want the
26:48
dimensions of my matri to be B comma number of heads comma number of tokens
26:53
comma head Dimension so what this will do is that you see the queries Matrix initi we grouped it with token one token
27:00
2 token 3 but now we'll group it with heads so now this is my head one and
27:06
this is my head two then what will happen is that within each head within each head there is this token one token
27:13
two and token three and each token now has head Dimensions which is equal to
27:18
three so this is three dimension similarly if you do if you look at head number two the first row of head number
27:25
two corresponds to token one token two and token 3 so you see the difference between this 1A 3A 2A 3 here the
27:32
grouping was with number of tokens but now we have grouped it with head one and head two why do we do this because it
27:39
just then easier to multiply right if you see uh if you see over here the
27:45
advantage of these heads is that the queries the keys and the values have been split into multiple heads so we
27:51
should clearly see the different copies right um now now that we have done this
27:57
type of a grouping we can clearly see the copies this is the first copy which is the queries this is
28:03
q1 this is q1 which is the queries for the first head this is Q2 which is the queries
28:10
Matrix for the second head so the division just becomes very easy whereas if you group it with the number of
28:16
tokens my head one is here my head one is here and my head one is here so part
28:23
of my head one is in this first row part is here and part is here so it needs to be grouped together in one single place
28:30
so that's why we group it with the number of heads because later remember that we have to then take the dot
28:36
product between so let's say this is my q1 now this is my Q2 now this is my K1
28:42
and this is the K2 we have to take a DOT product between q1 and K1 transpose and we have to take a DOT product between Q2
28:48
and K2 transpose remember this is exactly what we had done over here we had take we had taken a DOT product
28:55
between q1 and K1 transpose and we taken a DOT product between Q2 and K2
29:00
transpose to take this dot product all the uh to take this dot product
29:07
essentially head one needs to be in one place q1 needs to be in one place Q2 needs to be in one place K1 needs to be
29:13
in one place K2 needs to be one place so that's why it's very important for us to group The matrices by the number of
29:19
heads instead of grouping by the number of tokens and this is what is done in the next part of the code the next part
29:27
of the code we just take take the transpose of this these dimensions number one and dimensions number two so
29:33
keys. transpose 1 comma 2 just means that since python starts with zero indexing this is index zero this is
29:39
index one and this is index two so we are going to take the transpose of these index indexes these are going to be
29:45
interchanged to number of heads comma number of tokens so now we are going to be grouping by the number of heads so
29:52
all the keys queries and the values matrices are now transposed and we have 1 comma 2 here because one is the this
30:00
index and two is this index so these need to be interchanged now that's what's done in this part of the code
30:07
Okay so until now we have the q1 Q2 we have the K1 K2 and we have the V1 V2 so
30:14
if we were to map it out to the steps which we had seen in the previous lecture we have reached this part of the
30:20
code where we have q1 Q2 K1 K2 and V1 V2 now let's go to the next part where we
30:26
actually compute um where we actually compute the attention scores to compute the
30:33
attention scores what we have to Simply do is that we have to look at q1 and K1
30:39
and take their transpose we have to look at Q2 and K2 take their transpose that's it so that's exactly what we are doing
30:46
here we are going to take the queries and we are going to multiply it with keys. transpose 2 comma 3 why 2 comma 3
30:53
because now we are grouping by the number of heads right so when we look at one head
30:58
we essentially so first what we do is that we look at the first head so we have to multiply this with the transpose
31:05
of this Matrix so what does it mean taking the transpose of this Matrix so it means
31:12
that so now the rows here are T1 T2 and T3 right and the columns are the
31:17
dimensions so the rows here are the number of tokens and the columns are head Dimensions which means we have to
31:22
take the transpose of these two so multiplying K1 and multiply q1
31:28
and K1 transpose essentially just means taking the transpose of these last two Dimensions over here for K1 why the last
31:35
two because we are already so each row here corresponds to First token second
31:40
token third token and each column corresponds to the dimension which are essentially corresponding to the last
31:46
two Dimensions so multiplying K1 multiplying q1 with K1 transpose
31:51
essentially just means queries multiplies by keys. transpose 2A 3 so
31:56
now we have this entire queries made Matrix right when we take the squares Matrix and when we multiply it with
32:02
keys. transpose 2A 3 what will essentially happen is that first Q q1 will be multiplied with K1 transpose and
32:09
then Q2 will be multiplied with K2 transpose so that's essentially what we are going to get over here so if you
32:16
think about the dimensions of the resultant Matrix what we are now doing is
32:21
that uh we have this Matrix quaries which is B comma number of heads comma
32:27
number of Tok comma head Dimension and we are multiplying it with keys. transpose 2 comma 3 which means these
32:33
two are interchanged so we are multiplying this Matrix with B comma number of heads comma head dim comma
32:39
number of tokens right so what will the multiplication result in it's the number of tokens comma head dim multiplied by
32:46
the head dim comma number of tokens so it's going to be B comma number of heads comma number of tokens comma number of
32:52
tokens and that just going to be 3x3 so if you take take this
32:58
multiplication you'll get this Matrix which is of the size 1A 2 comma 3A 3 but
33:03
now what this means is that this is head one the these are this is actually head
33:08
one attention scores this is actually head one attention scores and this is actually a
33:15
had two attention scores and since these are attention
33:22
scores of course their Dimensions have to be number of tokens number of tokens
33:28
multiplied by the number of tokens so this is how we get the two
33:35
attention scores in matrix multiplication which is exactly what was done in the code also once we have the
33:41
sorry which was exactly what was done in our visual lecture once we have q1 and K1 we take this dot product once we have
33:48
Q2 and K2 we take this dot product so q1 multiplied by K1 transpose gives us the
33:53
head one attention scores Q2 multiplied by K2 transpose gives us this two attention scores but I want you to pay
34:00
very careful attention to the dimensions over here because the dimensions are where people usually get confused right
34:07
so you have this head one uh head 2 head 1 head 2 head 1 and head two so you have
34:12
q1 Q2 K1 K2 V1 V2 then what you have to do is that you have to multiply q1 with
34:17
K1 transpose Q2 with K2 transpose and when you do that you finally get this attention scores Matrix whose Dimensions
34:25
now are B which is the bat size is number of heads because you have grouped by the number of heads this head one and
34:31
this head two and why is this 3 comma 3 because since it's attention scores it
34:36
has to be number of tokens multiplied by number of tokens because the attention scores are calculated among every
34:43
token uh so this is now the two attention scores Matrix which we have for the two heads and this is exactly
34:50
what is done in the code also to get the attention scores we have to multiply the queries and keys. transpose 2 comma 3
34:57
this 2 comma 3 is very important because it's the last two Dimensions which get transposed and which get multiplied so
35:04
the ultimate di the ultimate dimensions of the attention scores after we take the dot product for each head is B comma
35:12
number of heads comma number of tokens comma number of tokens this step is also called taking the dot product for each
35:19
head why because first we multiply q1 with K1 transpose that's taking the dot
35:24
product for head number one and then we multiply Q 2 with K2 transpose that's
35:30
essentially taking the dot product for head number two Okay so until now we have found the attention scores Matrix
35:38
now what we have to do is that we have to find the attention weights so this is this is to get this what we had seen in
35:44
yesterday's lecture was to get the attention weights we have to basically first uh scale it then apply soft Max
35:52
then do causal attention and if needed we can do Dropout so now this is exactly what is
35:58
done in the mathematical calculations also so let me take you through
36:05
that okay so we have the attention scores M matrices now what we'll first do is that we'll first mask the
36:12
attention scores to implement causal attention so to do this so this is the
36:17
head one attention scores and these are the head two attention scores what we do is that the elements above the diagonal
36:23
are replaced with minus infinity we saw this in the causal attention lecture also so and the elements above the
36:29
diagonal in head number two are also replaced with minus infinity and uh what
36:34
we'll do is that we also divide by the square root of head Dimension remember in self attention we divided by the
36:40
square root of keys Dimension but now the keys Dimension is equal to the Head Dimension each key Dimension is equal to
36:47
the Head Dimension which is D out divided by number of heads which is 6 ided 2 so we'll scale it by the square
36:54
root of three we'll scale it by the square root of three and then we'll apply soft Max what soft Max will do is
37:01
that it will make sure the elements with negative Infinity are set to zero remember in causal attention we cannot
37:06
Peak into the future so for each token we only get the attention scores corresponding to that token and the
37:12
tokens which come before it and why do we divide by the square root of head Dimension this is just to make sure that
37:19
the variance of the query is multiplied by the keys transpose does not blow up
37:24
dividing by the square root of the head dimension make sure that the variance of
37:29
that dot product between queries and the keys transpose essentially
37:35
stays uh essentially stays closer to one and that's important for us when we are
37:41
going to do uh back propagation Etc we don't Valu we don't want values to be
37:47
widely different from each other so when we apply soft Max we get the attention weights Matrix and remember that the
37:54
dimensions of the attention weight Matrix are exactly same as the dimensions of the attention scores
37:59
matrix it's going to be batch size number of heads number of tokens and
38:05
number of tokens so the same thing here batch size one number of heads so this
38:12
is uh this is head number one this is head number two and then 3 comma 3
38:18
because I have number of uh tokens equal to three these number of rows and number of columns also equal to the number of
38:25
tokens number of tokens
38:30
but the difference now between the attention weights and the attention scores is that the attention score in
38:36
the attention weights if you see every row every row essentially sums up to
38:42
one so we can also Implement Dropout after this but I have not implemented it here for the sake of simplicity so now
38:49
what you can do is that you can go to the code and you'll see that the same thing has been implemented here first
38:55
what we do is that we create a mask of Nega negative Infinity above the diagonal which has been done over here
39:01
we create this mask of negative Infinity above the diagonal uh then what we do is
39:06
that we divide by the square root of the head Dimension uh and then we take the soft
39:12
Max and if needed we can also apply the Dropout so if you scroll up to the top
39:18
we can set the dropout rate by default I think the dropout rate we can set it to equal to zero if we don't want any
39:24
Dropout but if you randomly want to turn off certain attention weights you can do that by applying a dropout rate of let's
39:30
say 0. five okay so this is how the until now we have calculated the
39:36
attention weights and then apply Dropout then what we do after we get the attention
39:41
weights um remember the last step after getting the attention weights is that we have to
39:47
multiply um the head one we have to multiply the
39:53
head one attention weights with the value one V1 and we have to multiply the
39:59
head two attention weights with V2 so let's see how that is done now in matrix
40:04
multiplication um all right so this is the attentions
40:11
attention weight Matrix so this is head number one and this is head number
40:17
two and this these are my values Matrix right so my values is uh this is my
40:24
V1 and this is my this is my V V1 and this is my vs2 so H1 and H2 so what I'll
40:32
do is that I'll simply multiply these two together now take a look at the dimensions of what exactly is being
40:38
multiplied over here so so B comma number of heads and B
40:43
common number of heads that's the same for both these matrices or both these four dimensional matri they are both
40:49
grouped by the number of heads but what really we should check while multiplying is that this is number of tokens by
40:54
number of tokens so that's going to be 3x3 and this is number of tokens by the head Dimension so that's also 3x3 so
41:02
when you multiply this again the product is now taken into the number of tokens comma head Dimension space so we have
41:09
three tokens here and uh each head Dimension is equal to three right so when you multiply the
41:17
attention weights with the values you get the context Vector matrices so the first row over here is the context
41:23
Vector Matrix for head one and the second is the context Vector Matrix for head number two and we have three tokens
41:30
over here so there are context Vector for each tokens and the size of each context Vector is equal to the Head D is
41:36
equal to the Head Dimension which is equal to the number of which is equal to the last Dimension
41:42
over here which is equal to the Head Dimension now if you scroll to the visual multi-ad attention this is
41:48
exactly what we had obtained yesterday right we had obtained the head one context Matrix and we had obtained the
41:54
head two context Matrix here also there were 11 tokens and the size of each context Vector was equal to two which
42:00
was equal to the Head dimension in this case this is the same thing as what is been done over
42:07
here we have the context vectors for head one and we have the context vectors for head number two and when you go
42:13
inside each head the size is number of tokens and each token has the context
42:19
Vector of size equal to head Dimension this is done in this part where we multiply uh the attention we
42:27
multiplied with the values okay when we get the context Vector now what we do is
42:33
that when we get the context Vector remember that our final aim is not directly to get two different context
42:41
matrices but we have to merge the context Matrix for head one and the context Matrix for head 2 we have to
42:47
merge these context matrices right we we don't have to keep them separate so that
42:53
part is still remaining right and to merge these what we have to do is that we have to again Group by the number of
42:59
tokens so that's why we need to reshape it again currently the dimension is B comma number of heads right it's grouped
43:06
by the number of heads so we need to uh reshape it again remember we did this
43:11
step earlier once um where what here what we did is we actually uh switched it so we
43:19
deliberately brought the number of heads before so we want to group by the number of heads but now we'll switch it back to
43:24
the original configuration so that we group it by by the number of tokens so this is now token number one this is now
43:31
token number two and this is now token number three the reason I want to group it by tokens is that eventually I want
43:38
to merge the head one and the head two output for each token right so token one
43:43
it has the head one context vector and it has a head two context Vector I don't want it to be separate I want to merge
43:50
so then then I'll merge these two together similarly for token two I have
43:55
the head one context vector and I have the head two context Vector I don't want these vectors to be separate so I'll
44:00
merge these two for token number three I have the head one context vector and I
44:06
have the head two context Vector I don't want these to be separate so I'll merge these two and this merging is just
44:11
easier if I group it by the number of tokens so that's why we actually uh switch these positions once more and
44:19
that's the reason that there's one more transpose 1A 2 here so once we get the context Vector Matrix we'll again
44:25
transpose 1A 2 so that we'll group Again by the number of tokens and once we Group by the number
44:32
of tokens what we'll simply do is that we will merge for the token one we'll merge the first row and the second row
44:39
so it leads to six values which are the first two rows merged then for token two we'll merge these two head one context
44:47
Vector head 2 context Vector that will give me these six values and for token number three I'll merge these two
44:54
vectors so that will give me these six Valu Val again so ultimately the final
45:00
resultant context Vector Matrix which I have will be batch size number of tokens and the output Dimension so you see what
45:07
we did initially we started out with uh initially we started out with B comma number tokens comma D
45:14
in right then we went through a bunch of steps and then ultimately we obtained
45:20
the context Vector which is B comma number of tokens B comma number number of tokens
45:27
comma D out this is the final context Vector
45:32
Matrix and this is again the last part of my code the last part of my code is
45:38
this context. continuous. view what this will do is that this will just
45:44
merge um the first row second row of token one first row second row token two
45:49
first row second row of token number three and it will give me an output context Matrix of size 1A 3 comma 6
45:57
remember now the beauty of this context Vector Matrix is that whenever someone looks at this size they'll just see 1A
46:04
3A 6 but the way we have reached this is that we have OB we actually obtained two context vectors right we obtained two
46:11
context Vector matrices and then we merged them together into one so this
46:16
one final context Vector Matrix actually contains two perspectives it contains perspectives from the head one as well
46:23
as head two so it's much richer than having just the self ention mechanism producing a context Vector Matrix
46:30
because now we actually had multiple context Vector matrices and we merg them together if there were six attention
46:36
heads we would have six context Vector matrices which would be merged together to give me my final context Vector
46:42
Matrix that's the beauty of multi-ad attention although the dimension looks the same as it would have when we did
46:48
self attention but now the Matrix is much more richer since it captures multiple perspectives that's it this is
46:56
the last step of the multi-ad attention and I want to thank you all for sticking through this entire lecture and seeing
47:04
all the steps especially when we look at matrices and dimensions things can get a bit complicated and when you look at
47:10
this code directly you'll you'll think it's a bit complicated right but it's actually very simple if you understand
47:17
the mathematics with respect to matrices then the code actually makes a lot of sense this is the main class the
47:25
multi-ad attention which powers all the major large language models out there of course there were a lot of improvements
47:31
after this such as KV caching multi-head latent attention flash attention Etc but
47:37
if you understand these three dimensions B comma number of tokens comma D in if you understand what the keys. view
47:43
values. view queries. view does what transpose means transpose one comma 2
47:48
how it relates to the handwritten exercise you will you'll find these things are not very difficult so I
47:55
highly encourage all of you to take a piece of paper and write all these things down as as if you're following
48:01
this lecture seriously so once this class is actually defined you can what you can simply do
48:07
is that you can just take an inputs Vector which is 1 2 3 so I have two I
48:13
have three rows over here uh my three tokens and each token
48:19
is a six dimensional Vector the same example which we saw on the notes and then we can just have so here I'm having
48:26
two batches so I have stacked the inputs on top of the inputs so remember
48:31
although in the handwritten notes we just took one batch the code is powerful enough to change the First Dimension to
48:37
even two so that's what I'm considering over here I'm stacking these two inputs one on top of each other to create a
48:43
batch and then I just pass this entire input to this multi-ad attention that's
48:48
it and then I create the context vectors so you'll see the first context Vector
48:53
is 1x 3 comma 6 this is exactly the context Vector shape which we had seen over here um 1x3 comma 6 and the second
49:04
context Vector is 1x 3A 6 so this is the first batch this whole thing is the first batch and this whole thing is the
49:10
second batch that's why we have the size here 2x 3 comma 6 if we had three batches it would have been 3 comma 3
49:16
comma 6 so just within five to six lines of code we have implemented the multi-ad
49:22
attention uh calculation and here if you scroll above these are are 20 to 25
49:27
lines of code which is the mechanism which Powers uh or which is the brain behind
49:35
how why large language models work so well these 25 lines of code actually encode the key advancement which
49:42
happened in 2017 when the Transformer block was introduced for the first time and if
49:49
someone just takes a looks look at this code they'll find it difficult but my main purpose of today's class was to
49:55
link it to um handwritten notes of mathematical derivation and also to an
50:00
intuition which we looked at in the previous class only then I showed you the code so that you don't get scared or
50:06
intimidated by the code but if you're are seriously interested about developing your understanding and never
50:12
forgetting how multi-ad attention Works take a piece of paper write everything down so that you don't forget this at
50:18
all now that we have completed this lecture we have done multi-head attention so we have finished these
50:24
three part we are now fully ready to start learning about key value cache key
50:29
value cache is that main mechanism which made multihead attention much more efficient and this serves as the bridge
50:36
towards finally understanding multi-head latent attention that's the real key Innovation which was implemented in the
50:42
Deep seek paper but to understand key value cach and to understand multi-head latent attention it would have been very
50:48
difficult for you to understand this if you if you did not understand today's lecture so that's why we had all these
50:54
lectures on self attention causal attention multi-head attention so I want to congratulate you and thank you for
51:00
reaching this part please stay with me the later parts will be even more rewarding now that you have finished uh
51:08
completing the lectures until here so thanks a lot everyone please make notes along with me so that you learn the most
51:14
thanks everyone I look forward to seeing you in the next lecture









