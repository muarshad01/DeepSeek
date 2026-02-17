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
  
***

* 30:00
  
***

* 35:00

* Masking additional attention weights with dropout
* Dropout is a deep learning technique where randomly selected hidden layer units are ignored during training.
* This prevents overfitting and improves generalization performance.

***

* 40:00

***

* 45:00



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
















