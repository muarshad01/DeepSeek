#### Causal attention
* Causal attention, also known as a masked attention is a special form of self attention.

***

* 10:00

1. Causal attention, also known as a masked attention is a special form of self attention.
2. It restricts the model to only consider previous and current inputs in a sequence, when processing any given token.
3. This in contrast to self attention mechanism, which allows access to the entire input sequence at once.
4. When computing attention scores, the casual attention mechanism ensures that the model
5. To achieve this in GPT like LLMs, for each token processed, we mask out the futhre tokens, which come after the current token in the input text.

* Also called masked attention; unidirectional attention (autoregressive).

***

* 20:00

* We mask out the attention weights above the diagonal, and we normalize the non-masked attention weights, such that the attention weights sum upto 1 in each row.

***

* 25:00

* Attention score -> softmax -> attention weights -> Add 0's above diagonal -> masked attention score -> normalize rows -> masked attention weights
  
will be. 1. 5 0 0 0 this will be 0.05
25:32
2 2 0 0 Etc so the remaining two rows
25:37
also so now you see the problem here is that each row does not sum up to one now the first row sums up to 01 the second
25:44
row sums up to 6 Etc so what can you do in this case so that each row sums up to one the
25:51
simplest thing you can do is that the first row the entire summation is now 0.1 right so you divide this by 0.1
25:59
the second row entire summation is now 6 so you divide each element by
26:10
6 six the third row the entire summation is now 2 +24 + 0.05 which is 45 so you
26:18
divide each of these by 45 you divide this also by
26:26
045 and you divide this also by 045 what that will ensure is that that will
26:32
ensure that each um that will ensure that each row still sums up to one so
26:39
this is actually the first strategy which is implemented to uh to get the
26:45
causal attention scores the first strategy is like you have the you already have the attention scores to
26:50
which you have applied soft Max and you have got the attention weights you did that in self attention itself then what
26:55
you do is that you just add zeros above the the diagonals and you get the mass attention scores but then you normalize
27:02
the rows again so that they sum up to one and then ultimately you get the M attention weights that's the strategy
27:08
which we saw right now but do you see the problem with the strategy the problem with the strategy is that let's
27:14
see here again you see to go from attention scores to attention weights we are already doing soft Max so we are
27:20
doing one step of normalization here and then again we are doing one more step of normalization when we divide by the sum
27:26
of the rules so we are unnecessarily doing two normalization steps over here
27:32
so then the question is can we do a smarter way of normalization so that we
27:38
can we need to only apply softmax once and it turns out it is possible to do
27:43
that and the way to do that is we directly Target the attention scores so let me tell you how we do this so the
27:50
way to do that is we directly Target the attention scores here is where we make our intervention instead of making our
27:57
intervention in the tension weights so the way we do this now is that uh
28:03
essentially what we do is that all of these values all of these values which
28:09
I'm marking right now in circles which are essentially above the diagonal and
28:14
which need to be replaced eventually with zero and what we do with these values is
28:20
that we will replace these values with negative Infinity we'll replace these values with
28:26
negative Infinity so let let's see what the attention scores Matrix now becomes so the attention scores Matrix now
28:33
becomes 6 uh then negative infinity negative
28:39
infinity negative infinity and negative
28:48
Infinity uh the second row becomes
28:55
.1 um1 then
29:00
uh 1.8 and then again I will have
29:11
uh these three negative Infinity values so I just copy pasted them over here
29:17
then in the third row I'll have two negative Infinity values so I'll just show this for the three rows um and the
29:25
third row values will be 2 um the third row values will be
29:32
2 1.1 1.2 now why are we doing this because
29:38
remember that when you apply soft Max what soft Max does is that soft Max
29:44
actually takes the exponent So Soft Max will uh for the first row each element
29:51
the soft Max will replace with e to so the first element it will replace with e to X1 divided by the sum
29:59
the second element the soft Max will replace with e to
30:06
X2 divided by the sum so this we are doing for the first row uh so let's say let me Mark the
30:13
first rower here so if this is my first row each element will be replaced
30:19
with Mar this again so if this is my first row each element will be replaced with this e to X1 by su e to X2 by su
30:27
Etc now I want you to check what is e to minus infinity so you'll see that e to
30:34
minus infinity is actually equal to zero right so the second element will be
30:40
replaced with e to minus infinity / sum which is anyway zero because e to minus
30:46
infinity is zero this third element will be replaced with zero fourth element will be replaced with zero and Fifth
30:51
Element will be replac with zero essentially wherever we have minus infinity all of those will be replaced
30:56
by zero because X exponent ra to minus infinity is equal to 0 anyways and the first element will be
31:03
replaced with e to6 divided by E to6 so it will be
31:08
replaced with one so it will sum up to one here the first element will be replaced with e.1 divided e.1 +
31:17
e.8 the second element will be replaced with e to 1.8 divided e.1 plus e to 1.8
31:24
so the in each row will sum up to one and we'll also make make sure that all the elements above the diagonal are
31:30
essentially zero so this will make sure that we are not doing two stages of normalization here we did two stages of
31:37
normalization right we did soft Max followed by every row normalization but
31:42
here we are doing only one soft Max normalization that's it that's this trick of introducing negative Infinity
31:49
above the diagonal and it's a very powerful trick and uh it saves the computations for us so the more
31:56
efficient way is essentially you have the attention scores uh the more efficient way is that
32:03
you have uh these attention scores and then you apply something which is called as an upper triangular Infinity
32:11
mask um what this mask essentially means is that you replace the upper triangular
32:17
which is all the elements above the diagonal will be replaced with negative infinity and then you directly apply
32:23
soft Max only once so see here you have only one soft Max whereas here
32:28
uh you had one soft Max to get the attention weights and then you had another layer of normalization so that
32:34
way there are two normalizations so this on the other hand is a much more efficient way we are now just going to see this in
32:41
code so that you can understand uh uh what we are trying to do over here
32:46
remember in the previous lecture we started out with this inputs embedding Matrix where we had your journey starts
32:53
with one and step okay so these are the six inputs over here and there's a
32:59
vector embedding for each input here which is a three-dimensional Vector embedding and at the end of the previous
33:05
lecture we had defined this uh self attention class essentially what this
33:11
does is that it takes in the inputs it finds the keys queries and the values it
33:17
finds the attention scores then it gets the attention weights and then it finds the context Vector in causal attention
33:24
we are going to just make changes in this part so that all the elements about the diagonal are essentially masked out
33:30
right so let's see what is done in a causal attention so this section is titled hiding future words with causal
33:37
attention we start with the same inputs and I'm just printing out the attention weights over here and I'm first going to
33:44
show you this first approach in the first approach what we do is that we start with the attention weights which
33:50
have been already obtained previously so remember when we are showing these attention weights we have already
33:55
applied soft Max before to the attention scores and in the first in the first
34:01
method what we do is that we just take elements about the diagonal and put them to be equal to zero so now here is the
34:09
mask which is essentially all the elements above the diagonal are zero and then we apply this mask to the attention
34:16
weights so when we apply this mask to the attention weights you will see that we have the attention weights Matrix but
34:23
all the elements above the diagonal are now put to zero but this Pres presents the problem
34:28
that every row Now does not sum up to one right and that's an issue so to solve this what we do is that we simply
34:35
divide by the summation of the rows and this is what we saw uh this is exactly
34:41
what we saw actually on the white board over here if you remember um the first step what we saw
34:47
over here is you just put the elements above the diagonal to be zero and then just divide by the sum of the
34:55
rules uh so when you run this this will give you the mask attention weights and
35:01
this is the main purpose of causal attention now I'll show you the second method which is actually more effective
35:07
so here's the attention scores Matrix so remember in the second method we don't start with the attention weights like we
35:15
did in the first method we start with attention scores so we start with the attention scores and then here is the
35:22
mask which we have so we have this mask where there are ones above the diagonal
35:28
and then we take this mask and use it to replace all the elements above the diagonal with negative
35:34
infinity and once we have this then what we do is that we just have to take the soft Max once the soft Max will make
35:41
sure that all the negative Infinity over here are essentially put to zeros that's
35:46
what I'm showing on the screen right now and softmax will also ensure that the sum of every row is equal to one so now
35:54
if you actually compare these values so this is 1.55 17. 4483 and you'll see that these
36:03
are actually exactly the same values right the third row is 38309 7.31 03
36:10
38309 7.31 03 so essentially both the methods give us the exact same answer
36:16
but the second method is actually much much more effective because we just start with the attention scores uh we
36:23
replace the elements above the diagonal with negative infinity and we just take soft Max once we don't have to take the
36:30
soft Max and then again normalize as we did in the earlier method uh one more final step before we
36:37
actually move to coding the um causal attention mechanism is something called
36:44
as Dropout so Dropout is actually a deep learning technique and in this technique
36:49
what's done is that um sometimes during training neural networks we observe that
36:56
there are some neuron neur which don't learn anything and these neurons actually become lazy and they are not
37:04
contributing anything during the learning process so let's say if this my first neuron this is my second neuron
37:10
this is my third neuron this is my fourth neuron this is my
37:16
uh this is my fourth neuron this is my fifth
37:21
neuron and this is my six neuron
37:26
oops let's say this is my six neuron now let's say if these six neurons are
37:32
learning and we observe that out of these there are two neurons actually which are not doing anything which means
37:38
they are lazy neurons and all the work is being done by these four neurons one way to solve this problem is
37:45
that during the training process what we do is that we make sure that some neurons are randomly turned off so for
37:53
example if during the training process this neuron and this neuron are randomly turned off
38:00
and remember these were the neurons which used to do most of the work but now if these neurons are not there then
38:06
these two neurons have no other option but to learn something on their own right so Dropout actually solves this
38:13
issue of lazy neurons where some neurons don't do any work at all the fact that we randomly drop out or we make certain
38:20
neurons inactive other neurons have to pick up the pace it's like working on a group project right and if there are we
38:27
always have such groups in which there are only two people doing the work and others are not doing anything but if
38:32
those two people suddenly fall sick and they're not available the others have no
38:37
option but to work right it's the same case we drop out so what we actually do
38:42
is that in the case of causal attention a similar sort of a Dropout mechanism is applied where after calculation of the
38:51
attention weights let's say we get the attention weights Matrix and of course it will look something like this all the
38:57
elements above the diagonal are zero we randomly mask out certain attention weights we put them to zero randomly and
39:04
the and this is what do I mean random is that at every iteration different
39:09
weights are put to be equal to zero so the only thing which is fixed is the dropout rate so if the dropout rate is
39:16
50% it means that in every forward pass through the large language model 50% of
39:22
the attention um atten 50% of the weights in every row will be randomly put to zero
39:29
and the same weights will not be selected each time because the weights are selected in a random manner but on an average
39:36
half um of the attention weights will be put to zero that's what's done using the Dropout mechanism
39:43
so in this figure it actually illustrates that these gray uh these
39:48
gray neurons or rather these gray attention weights over here are masked out or they are randomly turned off and
39:56
the reason Dropout is implemented is that it improves generalization performance the issue with lazy neurons
40:02
is that if we are using this neural network on a new problem then the lazy neurons again won't fire and so
40:09
generalization will not be good so Dropout ensures that all neurons are effectively learning something and we
40:15
prevent overfitting to noise or we prevent um memorization of the data
40:24
which usually leads to overfitting and generalization issues CH so this is Dropout and let me just
40:31
show you in code how Dropout is implemented now so let's say if you have a matrix like this and imagine this is
40:37
your attention weights Matrix for the sake of Simplicity when you have a matrix like this and when you uh add a
40:43
Dropout so tor. nn. Dropout of 0.5 it means that on an average 50% of every
40:50
row is put to zero and remember this is on an average so it does not definitely mean that half of every row will be put
40:56
to Z Z so you see here none of these are essentially put to zero but here you
41:02
see five of them are put to zero here four of them are put to zero here three of them are put to zero so it's a random
41:09
process right but one thing is that in Dropout since we have a factor of 0.5
41:15
the values which are not put to zero are effectively scaled by a factor of two if we have a factor of point4 of Dropout
41:23
which means 40% it will be 1 divided 04 so every value will be multiplied by
41:30
that so to compensate for the reduction in active elements the values of the remaining elements in The Matrix are
41:36
scaled by a factor of 1 by5 which is 2 this scaling is crucial to maintain the
41:42
overall balance of the attention weights um ensuring that the average influence of the attention mechanism remains
41:48
consistent during the training and the inference phases so Dropout is a very simple mechanism you can think of
41:55
attention weights as a light bulbs right so if it's um if if these are the attention weights think of all of them
42:01
as light bulbs and Dropout is simply light bulbs going on and off during
42:07
every iteration and you can specify the dropout rate so if the dropout rate is
42:13
equal to uh if the dropout rate is equal to 50% which means that half of the
42:18
light bulbs in every row uh we will be randomly put to zero that is what is very important to
42:26
understand over here and this prevents overfitting and improve generalization
42:31
performance now that we have learned all of this we are actually ready to code the causal attention class uh we'll
42:37
start with the same input so we have six tokens over here your journey starts
42:43
with one step and you'll see that every token is a three-dimensional input
42:49
embedding Vector in this case what I'm going to do is that I'm going to create a two batches so this is the first batch
42:57
remember every batch has six tokens and every token has three dimensions right
43:02
so when I stack two batches on top of each other we'll have two which is the batch size six which are the number of
43:09
rows over here and three which are the number of columns that's essentially the input embedding Dimension
43:15
right so this is my input embedding Vector in causal attention the aim is the same we will ultimately take each
43:21
input embedding vector and convert it into a context Vector but the only thing which changes now is that let's say we
43:28
get these attention scores which are queries multiplied by the keys transpose the only thing which now
43:35
changes is that we replace all the elements above the token above the diagonal with negative infinity and then
43:42
we take the soft Max that's it actually so the attention
43:47
weights are calculated like this this will make sure that all the elements above the diagonal are equal to zero and
43:54
then what we do is that we apply the Dropout mechanism which means that in every row of the attention weights the
44:00
attention weights are actually plugged off and put to zero randomly this improves generalization as
44:07
we discussed earlier and the context Vector calculation then stays the same we just take the attention weights
44:13
Matrix and we multiply the multiply it with the values Matrix and we get the context Vector
44:19
Matrix so if you think of the difference between self attention and causal attention the only difference happens
44:26
after the attention scores are calculated after the attention scores are calculated we replace the elements
44:32
above the diagonal with negative Infinity we take a soft Max and then we apply Dropout so there are two changes
44:38
actually the first is the negative infinity and then soft Max and the second change is actually adding the
44:45
Dropout um which randomly puts off some attention weights to zero you might
44:50
notice certain things here which I have not explained such as what is this bias which is equal to false that essentially
44:57
means that we are just going to multiply the inputs with the keys quaries and the values right without any adding any bias
45:04
term that's why this bias is equal to false then secondly what is register
45:10
buffer over here so why is this mask being created with the self. register buffer so the main idea is it's not
45:17
strictly necessary but uh buffers are automatically move to the appropriate
45:22
device along with our model which will be relevant when training the llm lator this means that we don't need to
45:28
manually ensure that these tensors are on the same device as our model parameters which might avoid device
45:35
mismatch errors you don't need this right now per se but if it's there it's just a better
45:42
practice but remember these three dimensions are the ones which will show up throughout so B comma number of
45:48
tokens comma D in so every input has actually three dimensions which we saw over here the batch size the number of
45:56
tokens and input Dimensions that is very important to notice all right so now this caal
46:02
attention class is implemented so let's just test it out I'm going to assume D in equal to 3 and D out equal to 2 this
46:10
is the same uh input output Dimensions which we saw in this example if you
46:17
see um yeah just take a look at these Dimensions over here um let me scroll
46:25
over here yeah so the input Dimension is equal to 3 which is mentioned over here the input Dimension is equal to 3 but
46:32
you will see the output Dimension is equal to 2 so the queries keys and the values trainable weight Matrix uh
46:39
project every input Vector into a twood dimension output space as I mentioned in
46:44
the previous lecture in GPT and in modern llms the input and output dimensions are kept the same but here
46:51
let's just assume this for the sake of this example so then all we have to do
46:56
is that we have to define the uh batch and here remember the batch
47:02
is 2A 6A 3 and here if you
47:09
see uh yeah so the batch is 2 comma 6A 3 that's my input and then I just Define
47:16
the causal attention so causal attention class requires actually four um requires
47:22
four inputs so here I have my D in I have my D out I have my context length
47:27
and the final is the dropout rate so remember context length is batch. shape
47:33
one the first index so if you see bat do shape it's 2A 6 comma 3 so badge do
47:38
shape index by one will be six since I'm looking at six elements in the sequence the Contex length in this case is equal
47:44
to six okay uh so the context length is equal to six what else does the causal
47:51
attention class need it needs D in which is equal to 3 D out is equal to 2 context length is equal to 6 and it also
47:58
needs a dropout rate so here I have just mentioned dropout rate equal to zero so
48:03
you can run this and ultimately see this is uh yeah so here the dropout rate is
48:11
equal to zero so you have these attention weights before the Dropout
48:16
um and you have uh actually
48:22
this I think I should run this part once more so this is my Cal ition class D in
48:28
D in um yeah so now that I print the context
48:34
vectors these are the context vectors which I obtained right so this is size 6x2 this is exactly the size which we
48:42
saw uh here so the context vectors will have a size of 6x2 which are six rows
48:47
and two columns but now remember that we have two batches right so in the first batch we have six tokens which will be
48:55
processed and that will lead to a context Vector Matrix of 6x2 when you
49:00
have the second batch that will lead to another context Vector Matrix of
49:05
6x2 um so that's the second batch and now when both remember we are passing
49:11
two batches here so if you see the input we are stacking two
49:17
batches here so there are two batches so the output will be 2A 6A 2 so here is 2A
49:24
6 the first batch output the first batch context Vector Matrix and this is the
49:30
second batch context Vector Matrix so the output size is 2A 6A 2 so don't be
49:35
confused that why is the output size or output Dimensions not equal to 6x2 the
49:40
reason it's 2A 6A 2 is that we have two batches and each has a size 6x2 right so this is my context Vector
49:48
Matrix and it seems to be correct uh here I have just written another function which prints out the attention
49:55
weights before Dropout and after after Dropout so here I have run the causal
50:00
attention class with the dropout rate equal to.5 and uh I have shown the prints
50:06
before Dropout and after Dropout so you'll see that before Dropout let's see here there were four two weights which
50:12
were active right but after Dropout only one of this is active but this is scaled by a factor of two because dropout rate
50:19
is now equal to 0.5 similarly if you look in the third row there were three
50:25
weights which were active before dro out but after after Dropout only two are active now if you look in the fifth row
50:32
there were five weights which were active before Dropout and after Dropout none of them are active remember it's a
50:38
random process so on an average 50% of the weights will be turned off but it might happen that sometimes all of the
50:44
weights are turned off so you can run this part to check for yourself how the
50:49
Dropout actually works this brings us to the end of the lecture on causal attention remember
50:56
remember causal attention is actually quite simple if you understand the self attention mechanism the main intuition
51:03
behind causal attention is that we cannot look into the future for a given token we only have access to that token
51:09
and the information which precedes it so what we actually do is that we need to take the attention weights and set all
51:15
the elements above the diagonal to be equal to zero and there are two ways of actually doing this the first way is you
51:22
start with the attention weights directly which were computed before four from the attention scores and soft Max
51:29
and you put all the elements to above the diagonal to zero and then you do the normalization of the rows once more but
51:35
here you have to do soft Max once and then you again have to do this normalization of the rows so that's two
51:41
times normalization instead we can make this process a lot more efficient by just starting from the attention scores
51:48
Matrix before we applied the soft Max and then put all the elements above the diagonal to be equal to infinity and
51:55
then when you calculate the soft Max it will automatically ensure that all the elements with negative Infinity are set
52:00
to zero and every row every row essentially sums up to one so that's how
52:06
you get the attention weights and that's how you make their elements above the diagonal to be equal to zero but in
52:12
causal attention we don't stop there we usually even add Dropout so certain
52:18
weights are randomly mased out and set to zero to improve the generalization performance and different weights are
52:24
selected in every form forward pass so this makes sure that there are no lazy
52:29
weights and every every weight essentially is learning something during the training process and then we
52:35
implemented a causal attention class which actually looks like this if you see it's like 15 to 20 lines of code and
52:42
I hope you have understood every single aspect of this code over here remember these three dimensions of the input
52:48
shape which is the batch size um the number of tokens and the input
52:53
dimensions as you must have noticed in every lecture I pay a lot of attention to
53:00
Dimensions because ultimately I think that understanding attention really comes down to matrices and people are
53:07
not really comfortable with matrices because they cannot visualize the dimensions that's why I pay a lot of
53:13
attention to dimensions in the next class what we'll be doing is that we'll be advancing
53:18
further until now we have finished looking at self attention we have finished looking at causal attention in
53:24
the next class we'll start looking at multi-ad attention then we will be completely ready to understand key value
53:31
cach and finally we'll be fully ready to understand the multi-head latent attention as I mention always always try
53:38
to make notes as I'm explaining so that you fully understand what's happening in the lecture if you just listen to the
53:44
lecture you will feel as if you're understanding but the concepts will not be strengthened so the lectures are
53:51
going to get deeper and deeper now as we proceed into the further modules and I really want you to stick with me through
53:57
this course and finish and finish all the lectures so stay motivated keep making notes and ask doubts so we'll be
54:04
able to clarify them thanks a lot everyone and I look forward to seeing you in the next lecture











