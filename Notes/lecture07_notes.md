#### problem with the self attention mechanism is and
* The artist painted the portrait of a woman with a brush.
1. painting a woman with a "brush"
2. painting of a "woman with a brush"

***

* 10:00

* Self attention can only capture a single perspective in a given input sequence. It cann't capture multiple perspectives.

***

* 20:00

| Self Attention | Perspective |
|---|---|
| 1 Self Attention | 1 Perspective | 1 Context Vector Matrix | 
| Multiple Self Attention | Multiple Perspective | Multiple Context Vector Matrices |

***

* 25:00

#### Implementing a 2-head attention (step-by-step)

1. Input Embedding
* Example: (11 X 8), where d_in=8
2. Start with a single W_q, W_k, W_v
* Example: (8 X 4), where d_out=t
* Output is (11x4) for query vectors, key vectors, value vectors
3. Split W_q, W_k, W_v into multiple heads
* Example: (8 X 2) (W_q1,W_q2), (W_k1,W_k2), (W_v1,W_v2)

$$\text{head-dim} = \frac{d_{out}}{num ~of ~heads}$$

***

literature and in other videos it just explained in a complicated and reverse manner instead it's much easier to
25:13
explain multi-ad attention if you motivate it like this and then if you show the step by- step uh visual Matrix
25:20
or visual matrices calculation so let's see the step by- step procedure right now okay the first thing which we do is
25:28
as always we start with an input embedding Matrix and the input embedding Matrix looks like this we have these
25:35
Tokens The Artist painted the portrait of a woman with a brush right we have
25:42
these 11 tokens and then every token is essentially uh an input embedding of
25:49
eight Dimensions which we have considered over here this is also called as the input embedding Dimension or D in
25:55
so the dimensions of this entire Matrix are we have 11 rows and we have eight
26:00
columns so the dimensions are 11 by 8 that's my input embedding Matrix
26:05
remember the goal of this two-head attention now is to take this input embedding Matrix and to convert it into
26:12
two context uh context Vector matrices not just one we have to convert it into two
26:17
so that each context Vector Matrix captures a different perspective so this is the input
26:23
embedding Matrix which we have started with and then the first thing which I want to do show you what we would have
26:29
done if we just had a single uh single attention head so if we have a single
26:35
attention head what would we have done we would have multiplied this input embedding Matrix with the trainable
26:41
query Matrix which is an 8x4 Dimension Matrix a trainable key Matrix which is an 8x4 Dimension Matrix and a trainable
26:48
value Matrix which is 8x4 and this multiplication would have resulted in the query Vector Matrix that's 11 by4 a
26:55
key Vector Matrix that's 11 by4 and a value Vector Matrix that's 11 by4 now to
27:01
extend this into a multi-ad attention or two head attention in this case what we have to do is that we have to first
27:07
decide on the output Dimension which we want and here I'm deciding the output Dimension is equal to four that is
27:14
something which is fixed at the start the input Dimension is fixed the output Dimension is fixed then what we do is
27:21
that uh then we decide the number of heads which we want and here we are
27:26
having two attention heads right so this output Dimension is then split among these two attention heads so each
27:33
attention head will have Dimension equal to two and the way this visually looks right now is something like this so now
27:40
my trainable query key and the value weight matrices earlier they looked like this right but now I'm just going to
27:48
divide them into two parts so my trainable query Matrix is now for the first attention head it's this so see
27:55
the size of this has been now it's eight row and two columns instead of eight rows and four columns and for my second
28:02
head the trainable query Matrix is this which is again eight rows and two columns so in terms of nomenclature now
28:10
instead of WQ I have wq1 which corresponds to the first head and I have
28:15
wq2 which corresponds to the second head so if Dimensions understanding
28:20
Dimensions is a bit difficult just remember that to split this into two attention heads I've just divided the
28:26
query into two parts the keys weight Matrix into two and the value weight Matrix into two that's
28:33
it so this Dimension over here which the trainable query weight Matrix has the
28:39
trainable query second uh for the second head this is called as the head
28:44
Dimension and the head Dimension is just the output Dimension divided by the number of heads so the output dimension
28:51
in our case is equal to 4 and the number of heads is two so the head Dimension is just 4X 2 which is equal to 2
28:58
so what is the head Dimension it's essentially the number of columns in each attention head and that's equal to
29:04
two in our case and the same split happens for the trainable key Matrix and
29:10
the trainable value Matrix as well so similar to what we did for the trainable
29:15
query Matrix we now have W K1 W V1 W K2
29:21
and W V2 so these are the trainable key matrices for the both heads
29:28
and these are the trainable value matrices for the both heads so see what we are doing in this step is that we are
29:34
creating multiple copies of the trainable uh query weight Matrix the
29:39
trainable key weight Matrix and the trainable value weight Matrix this is the main idea in the multi-ad attention
29:45
and if you think about it it's quite simple right uh in a single head we just had one Matrix and remember that now
29:52
that the D out is fixed we cannot change this so if we want two attention heads we just split the D out out into two
29:59
parts so this is my these are my trainable query weight matrices trainable key we matrices and trainable
30:06
value at matrices that's step number three we essentially split or create
30:11
multiple copies of WQ WK and WV now now that we have multiple copies of WK WQ WK
30:19
and WV it will naturally create multiple copies of the query vectors the key vectors and the value vectors right
30:26
because uh let's look at the query vectors first I will first take my input embedding Matrix
30:33
X and I will multiply it with wq1 so that's 11 by 8 * 8x2 and that will give
30:39
me my first query Vector Matrix q1 which is 11 by2 then I will take my input
30:45
embedding Matrix X and multiply it with wq2 that will give me my second query
30:51
Vector Matrix that's Q2 which is again 11 by2 similarly I take my input
30:56
embedding Matrix multiply with W K1 and W K2 and that gives me the two key
31:02

***



Vector matrices and I take my input embedding Matrix and I multiply it with W V1 and I
31:09
multiply it with W V2 and then I get my two value Vector matrices V1 and V2 now
31:16
remember here what we have done simply is that instead of having one one query Vector Matrix one key
31:23
Vector Matrix and one value Vector Matrix for single head since we have multiple heads now we have two query
31:30
Vector matrices q1 and Q2 we have two key Vector matrices K1 and K2 and we
31:35
have two value Vector matrices V1 and V2 and what are the dimensions of this
31:41
these uh these matrices the number of rows essentially remain the same so if
31:47
you see for all of them the number of rows remains 11 why do the number of rows remain 11
31:53
because the number of tokens which I have the artist painted the portrait of a woman with a brush those are 11 tokens
32:00
but the key thing to note here is that the number of columns which we have the number of columns now becomes equal to
32:06
two because that's the head Dimension remember the head Dimension is just the D out divided by the number of heads
32:14
which is equal to 4 / 2 which is equal to 2 so the number of columns in all of these matrices are equal to two again if
32:22
you are getting confused just look at the head number one all of these matrices are the query key and the value
32:27
matri es for head number one and all of these matrices are the query the key and
32:32
the value matrices for head number two remember this head number one we
32:38
have these matrices and head number two we have these matrices it just that we have now created multiple copies so what
32:45
happens is that we have the same vectors for a single head but now we have two copies and now that we have two copies
32:51
we still have only four dimensions right so each copy has to have only two Dimensions remember the D out is fixed
32:58
at the start so in Step number four we create multiple copies of the query the key and
33:04
the value Vector matrices which I have denoted over here right now q1 Q2 K1 K2
33:10
V1 V2 now think about this right what is usually done in the next step usually we
33:15
take the dot product of queries and the keys transpose to get the attention score Matrix but here we have two query
33:23
matrices we have two key matrices so what will happen naturally we will have two attention score matrices right so
33:30
that's what happen next we compute the attention scores for each attention head
33:35
so this is q1 Q2 K1 and K2 so for computing the head one attention scores
33:42
what we simply do is we multiply q1 with the we multiply
33:48
q1 uh with K1 transpose we multiply q1 with
33:53
K1 transpose so that's 11 by 2 * 2A 11 and and that gives us the attention
33:59
score of the first head that's 11 by 11 then what we do is that to find the
34:04
attention scores Matrix of the second head we multiply Q2 with K2 transpose so
34:10
that's 11x 2 * 2x 11 that's 11 by 11 so now take a look here what is exactly
34:16
happening when we looked at uh single head if we look at a single head
34:22
attention we'll have an attention score Matrix of 11 by 11 right if you just look at one head because there are 11
34:29
tokens here the cool thing which has happened or the amazing thing which has happened with multiple heads is that
34:35
although the output Dimension is getting split into two parts so the head Dimension is equal to two the attention
34:41
scores Dimension Remains the Same for both the heads it's 11 by 11 for the
34:46
first head and it's 11 by 11 for the second head and it would have been 11 by 11 if we just did a single head so
34:53
essentially now what we have done is that we have two copies of the attention scores we have 1 11 by 11 attention
34:59
score and 1 11 by1 attention score why is it 11 by1 because remember there are
35:05
11 tokens right the artist painted the woman painted the portrait of a woman
35:11
with a brush Etc and if you think about where we started with this is exactly what we
35:16
wanted right instead of just getting one attention scores Matrix we wanted to
35:21
extend the self attention mechanism so that we can get multiple attention scores matrices and that is exactly what
35:28
is happened here since we had two copies of the queries since we had two copies
35:33
of the queries and we had two copies of the keys we can essentially multiply these two copies and get two attention
35:39
scores matrices so each attention score Matrix essentially can capture a different perspective and that's the
35:46
main advantage of multi-ad attention this step here that although we have multiple heads and although the
35:52
dimension of each head is now split into two so the head Dimension is now equal
35:58
to 2 which was 4 before so in single head attention it was 11x 4 * 4X 11 and
36:05
that gave us the 11 by 11 attention score Matrix but now it's 11x 2
36:10
multiplied by 2x 11 so although this Dimension is reduced by half so although this Dimension is
36:17
essentially reduced by half the final attention score Matrix still is 11 by 11
36:23
so this Dimension is same in both of these cases that's the beauty of multi-head
36:28
attention although the head each head has a reduced Dimension when we take the
36:33
dot product of the queries and keys transpose for both the heads it's we get two attention score matrices of
36:40
Dimensions 11 by 11 and each of these can now capture a different perspective essentially we have two copies of the
36:46
attention score matrices now then what happens in the next step is the same
36:52
since we have two copies of the attention scores now what we'll do is that we'll scale we'll scale by square
36:58
root of the keys Dimension we'll apply soft Max and then we'll apply causal attention which means that we'll just
37:04
make sure that all the elements above the diagonal are set to zero remember we cannot peek into the future and if
37:10
needed we can also apply Dropout so in this schematic I've assumed the dropout rate to be zero but after you get the
37:18
attention weight Matrix you can even had have a dropout rate and randomly turn off different elements in the attention
37:25
weight Matrix so this is the head one attention weight Matrix that is 11 by 11
37:30
and this is the head two attention weight Matrix that is also 11 by 11 Matrix now and what is the difference
37:37
between attention weights and attention scores attention weights every row will just be normalized so if you look at
37:43
every row it will be summed up to one and also remember that we are implementing causality here so we make
37:50
sure that for both of these attention weight matrices the elements above the diagonal will essentially be equal to
37:57
zero so just keep that in mind and then what
38:02
we do in the last step is that now we have two we have an attention head Matrix for both these heads right and
38:08
remember earlier we had calculated the value matrices V1 and vs2 V1 was 11 by2
38:14
V2 was 11 by2 so V1 is the value Matrix for head 1 V2 is the value Matrix for
38:20
head 2 so what we will do in this last step is essentially we take the
38:25
attention weight Matrix of the first head we multiply it with the value Matrix of the first head so that gives
38:31
us the context Vector Matrix for head 1 which is 11 by2 and for the second head
38:36
we similarly take the attention weight matri of the second head and we multiply it with the value value Vector for the
38:45
second head so that's 11 by 11 * 11 by2 and that gives us the head two context
38:50
Matrix so head 1 context Matrix is 11 by2 and head 2 context Matrix is also 11
38:56
by2 and now remember what we do after this point is that we have the context Vector
39:03
matrices from both the heads and remember what we had discussed at the start once we have the context uh once
39:10
we have the context Vector Matrix for the head number one and once we have the context Vector Matrix for head number
39:16
two we'll just merge these context Vector matrices and that's exactly what we do in the last step in the last step
39:23
what we do is that we have the first head one context Matrix which is which you you can say as giving us the first
39:30
perspective that is perspective one and we have the head sorry this
39:35
should be head two so we have the head two context Matrix and that essentially gives us the perspective
39:41
two and when you merge these context Vector matrices you will have the final context Vector Matrix which is of the
39:48
size of 11x 4 so to the left side of this is my first head to the right side
39:55
of this is my second head so ultimately when I merge the results from both the heads together I'll have the context
40:02
meor of vector of size 11 by4 and remember now if you had just done a single head attention if you had just
40:09
done a single head attention uh without splitting into two heads the output Dimension is four right so you would
40:15
have also got the same context Vector Matrix size 11x 4 but it would not have
40:21
consisted of two perspectives in a single head if you had just used a single head there also you would have
40:27
have gotten the same context Vector Matrix of 11x 4 but there the whole
40:32
thing would have been just one perspective but now the advantage here is that the final size Remains the Same
40:39
but it consists of two perspectives the first perspective given by my first head
40:44
which I'm calling P1 and the second perspective given by my second head which is called as P2 so we have just
40:50
extracted more information from my text of course the disadvantage of this is that for extracting each perspective we
40:58
have only two Dimensions to play with now that's the drawback whereas here we had essentially four dimensions for each
41:04
perspective right but now we have reduced number of Dimensions to play with for each perspective that's the
41:10
drawback for multi-ad attention the main drawback is that the dimension size uh
41:16
for each head reduces right as you see over here the the dimension for each
41:21
head is effectively reduced because we have to split the whole query weight Matrix key we Matrix and value with
41:28
Matrix into two so the dimension size for each head is reduced so the amount of information we can capture is a bit
41:34
reduced but the number of perspectives we can capture is is increased so each head captures more perspective so the
41:42
way I think about it is like divide and conquer instead of Conquering the whole sentence at once you divide into
41:48
different parts and then each part conquer some different perspective um that's the simplest way I like to think
41:54
about multi-ad attention so this whole stepbystep procedure which
42:00
we saw let's recap it quickly we start with the input embedding Matrix the artist painted the portrait of a woman
42:06
with a brush and I've deliberately started with the sentence here which can be looked at from different perspectives
42:12
correct so what we do here is that we start with the input embedding Matrix and when we multiply it with the
42:19
trainable query key and the value Matrix we split these trainable weight matrices into two parts so we fix the outut
42:27
Dimension equal to four and we decide the number of heads so since we have two heads here each head will essentially
42:33
get two dimensions that's called as the head Dimension which is the D out 4
42:38
divided by the number of heads which is equal to 2 so wq1 is 8 by2 wq2 is 8 by2
42:44
Etc so we have this these are the query key and the value trainable weight matri
42:49
for head number one and these are the trainable query key and the value we mates for head number
42:56
two all right so once we have these multiple copies of w q WK and WV
43:02
naturally it leads to multiple copies of query key and value so head 1 has one
43:07
copy of Q KV which is q1 k1 and V1 and head 2 has another copy of qkv that's Q2
43:13
K2 and V2 then what we do is that for q1 and K1 we have the first attention
43:20
scores Matrix for Q2 and K2 we have the second attention scores Matrix the first
43:25
attention scores Matrix is from head one second attention score Matrix is from head two why do we have two attention
43:32
score matrices well each head might be capturing a different perspective such as maybe the first head might be
43:38
capturing this perspective maybe the second head might be capturing this perspective Etc so each head might be
43:44
capturing different perspective and that's why we have two uh attention
43:49
scores Matrix here in my view this part is the most important step because here
43:55
we see that each attention scores Matrix captures a different perspective and that's the whole advantage of the
44:00
multi-head attention mechanism and then after that we follow similar steps which we had seen for self attention we then
44:08
take the attention score Matrix scale it by square root of keys Dimension apply soft Max apply causal attention which
44:14
means we mask out all elements above the diagonal in the attention weights to be zero and then if needed we can apply
44:21
Dropout to improve the generalization performance or to prevent overfitting so until this point we have the attention
44:27
weights which have been calculated and then what we do is we multiply the attention weights for every head into
44:34
the value Vector for that head V1 and V2 and then we get the context Vector Matrix for head number one context
44:40
Vector Matrix for head number two what the context Vector Matrix for each head represents is that now we have 11 rows
44:48
here right the artist painted Etc so we go from input embedding to a
44:53
context Vector so now for artist instead of just looking looking at the semantic notion of artist the
44:59
context Vector for artist now captures information about how this artist relates to the other tokens that's why
45:06
this Matrix is much more richer than the input embedding Matrix so this is the head one context Vector Matrix and this
45:12
is the head two context Vector Matrix and in the last step what we do is that we merge the context Vector matrices for
45:18
both the heads and that leads to the final context Vector Matrix 11x 4 the size of this is the same as what it
45:25
would have been if we just used self attention with a single head but the main advantage is that we have now two
45:31
perspectives within this context Vector Matrix P1 and P2 so hopefully we'll capture richer representations in the
45:39
text itself the disadvantage is of course in each perspective we now get reduced number of Dimensions to play
45:45
with so the expressivity in each perspective might be reduced but this is a trade-off which
45:51
seems to work well in our favor because all the modern llms are based on the
45:57
multi-ad attention mechanism we no llm just has a single head we have multiple heads so that each head can capture
46:03
different perspective so this is the whole stepbystep procedure of how we go from the self attention mechanism to the
46:11
multi-head attention mechanism which was the main purpose of today's lecture now what I want to show you is that I want
46:17
to show you a very quick demonstration of uh how of visualization of these
46:24
attention heads so what we are going to do is that we we are going to take a pre-trained large language
46:30
model we are going to pre we are going to take a pre-trained llm so this is going to be a Burt model and it will
46:37
have a bod directional attention so causality is not implemented so every token will look at previous tokens and
46:43
also the tokens after that so this is pre-trained which means it has already been optimized on a huge amount of data
46:50
what we'll do is that we'll pass our input sentence we'll pass our input sentence to this pre-trend llm and
46:56
what's the input sentence the artist painted the portrait of a woman with a brush we'll pass our input sentence to
47:02
this pre-rain llm and then what we'll do is that we'll peek into the different
47:08
uh attention heads and we'll see what every attention head essentially gives
47:13
us so remember that when we see the code there will be two parameters there will be layer there will be layer and the
47:20
head so what layer essentially means is that an llm architecture has multiple
47:26
Transformer blocks and each Transformer block has multiple attention heads so when we look at different layers it
47:33
means different the different Transformer block and when we look at head it means that which head we are in
47:39
in a particular layer so for the purposes of demonstration we are only going to look at layer number three
47:46
which is essentially the third Transformer block and in this layer number three we are going to look at
47:51
attention head number three and attention head number eight what does it mean attention head number three and
47:57
attention head number eight the pre-trend llm which we are looking at we have 11 attention heads so the output
48:04
Dimension is split into 11 different parts and then every every attention head will essentially get D out divided
48:11
by 11 and then we are going to look at the attention weights Matrix such as
48:17
this uh such as this for each head and that is going to tell us that when we
48:24
look at wom for example what is prioritized by different attention heads
48:29
so let's quickly jump into the demonstration right now so the uh package which I've download downloaded
48:36
over here is bwiz and then what I'm simply doing is that I've loaded the pre-trained uh
48:43
pre-trend model over here and I'm just showing a visualization for this sentence the artist painted the portrait
48:49
of a woman with a brush and first I want to show you for layer number three and
48:54
we'll see for layer number three and essentially head number three so if you go into layer number three and head
49:00
number three and if you hover on to woman let's see so if you hover on to woman you will see that the maximum
49:06
attention is given to brush if you see on the right hand side the maximum attention if you trace this line You'll
49:13
see that the maximum attention is given to brush and we can also confirm this so
49:19
um in this in this code I've essentially plotted the different uh attention
49:25
scores which are given for or woman and I've have taken a screenshot over here
49:30
so if you look at layer three and head number three and if you take the query as woman here you can plot the tokens
49:39
for which the maximum attention weight is given so the maximum attention weight or score is given to brush for layer
49:45
three and head three but now let's go to layer three and head number eight so if I go to layer three and head number
49:51
eight right now you'll see that when you see wom the maximum attention is now given to Port
49:57
and that is again confirmed over here if you see layer number three and head number eight maximum attention is given
50:03
to Portrait so this might indicate that here the attention of the woman is given
50:09
to brush right so that might mean that here we have the second
50:14
visualization um so let me take a
50:24
oops um after this loads let me take a screenshot of uh that second
50:30
visualization yeah so it it seems that the attention between woman and brush is
50:35
the maximum right so it seems that head number three which we saw over here thinks of this interpretation or this
50:42
perspective because it seems in the second perspective the woman holds a brush in her hand and the attention
50:48
between woman and brush is the maximum in this head whereas if you look at head number eight the attention between woman
50:55
and brush is very low so it seems that this head has recognized that maybe the woman is not holding the brush but the
51:01
woman is just present in the portrait so it might mean that this second head thinks that this
51:08
perspective uh is more strong so it it maybe decodes this
51:16
perspective so here you can see this is the direct proof that a pre-trained Transformer or pre-trained llm rather
51:23
has different attention heads and each attention head can essentially uncover a different meaning the the head number
51:30
three which we saw over here uncovers this meaning that maybe the woman holds a brush in her hand whereas head number
51:37
eight uncovers this meaning that maybe it's just a portrait of a woman and the woman might not be holding a brush but
51:43
the artist might just be painting the portrait of a woman so in this Hands-On demonstration we just saw that different
51:50
attention heads can capture different perspectives and that's the whole aim or the whole purpose of the multi-head
51:56
attention mechanism I hope I've been able to explain why what's the intuitive need
52:03
that we need to go from the self attention mechanism to the multi-head attention mechanism this lecture was
52:08
specifically dedicated to introducing the intuition behind multihead attention mechanism in the next lecture what we
52:15
are going to do is we are going to do actual calculation using mathematical numbers so we are going to start with a
52:21
given sentence we are going to assume some numbers and we are going to uh apply multihead attention in practice
52:28
but I did not want to directly jump to this lecture without giving you an intuition for why we move from self
52:33
attention mechanism to multi-ad attention mechanism this is the core building block of why llms work and it's
52:41
also the core building block which deep seek figured out that we need to modify this block itself uh to make it a bit
52:48
better so although multi-ad attention has a lot of advantages it does have some disadvantages in some in terms of
52:54
storage space and comput efficiency which are mitigated or reduced by key
52:59
value cash and further reduced by multihead latent attention so that's why we have this important milestone in our
53:05
way we cannot understand KV cach or multi-head latent attention without understanding multi-ad attention itself
53:13
so after the next lecture we'll directly move to Key value cach and then multi-ad latent attention which is the First
53:20
Fundamental innovation in deep seek so stay tuned till that time and I hope you are making notes along side these
53:27
lectures are a bit dense and I'm deliberately making them a bit longer so
53:32
that everything is explained to you this won't be a lecture series of two to three lectures I'm planning it to make
53:38
it 35 to 40 videos of lecture Series so that ultimately you really understand
53:44
the nuts and bols of how deep seek is constructed but to go to that stage it's
53:49
important for us to be on the same page with the building blocks thanks a lot everyone and I look forward to seeing
53:55
you in the next lecture










