#### Code Mixture of Experts (MoE) from Scratch in Python
* [makeMoE: Implement a Sparse Mixture of Experts Language Model from Scratch](https://huggingface.co/blog/AviSoori1x/makemoe-from-scratch)
* [minGPT](https://github.com/karpathy/minGPT)

***

* 5:00

* T4 GPU
* A100 GPU

* __Activation Function__: ReLu

* __Step 2__: Implement the Router

***

* 10:00


that gives us the expert selector matrix. So the router determines which
10:12
um the router determines which expert network receives the output um for each token from the multi head
10:19
attention. Let's see how we will implement this. First I'm going to select the number of experts and we are
10:25
going to use the same number of experts as in the figure over here. So the number of experts I'm going to set to be
10:31
equal to three and top K this is this means that every token will be routed to
10:37


***


the top two experts out of these three. We'll see that a bit later. First I'm going to just see the routing and the
10:43
embedding dimension which we are going to use is equal to 8. Okay. Uh then the output which is the
10:51
output from the multiheld attention which is also the input to the mixture of experts module. We are going to
10:57
assume batch size of one uh number of tokens equal to four and the embedding dimension is equal to 8. Okay. So the
11:05
input matrix which is shown over here in the figure is the image output in the code and this top kg gate linear that's
11:13
my router. So if you take a closer look at the router and its dimensions
11:19
the the first dimension over here is the embedding dimension and the second dimension is the number of experts.
11:26
That's exactly what we have written over here. The first dimension is the embedding dimension and the second dimension is the number of
11:33
experts. And then what we are going to do is we are going to multiply the input matrix with the output out with the uh
11:40
routing matrix. This is exactly what is done here. The input to the multi the input to the mixture of experts is
11:47
multiplied essentially with this uh routing matrix and then we get the logit matrix. So what I have called as expert
11:54
selector matrix in the code that is also called as the logits matrix. All right.
11:59
So then you can run this and then you'll get the logits matrix which is essentially I have four rows because
12:06
every row corresponds to one token and I have three columns because every column corresponds to a particular
12:12
expert. Right? So we have this expert selector matrix which I have shown over here is also called as the logits matrix
12:19
in the code. So until now we have got the expert selector matrix but we have not
12:25
implemented the load balancing which means now I want to make sure that every token is only routed to two experts
12:33
right I want to make sure every token is routed to two experts and I also want to make sure that the weighted sum uh for
12:41
all the experts a given token is routed to that sums up to one so let's implement these two things first we are
12:47
going to implement the top k load balancing Right? For this we simply need to do something like this. Right? In
12:54
every row here I only want to select two experts now. So to do that we we take
13:00
the logits matrix which is also the which is also the expert selector matrix and we do logits do topk. Top k is a
13:07
functionality of pytorch which just selects the top k dimensions uh from every row. And
13:15
remember we have defined top k to be equal to two. So you can run this and you'll get two things. You'll get the
13:21
top K logitics and you'll get the top K indexes. What top K indexes means is that for every rows uh which indexes are
13:28
chosen. So for the first row we choose the zero index and the first index which is 0.0238 and minus
13:36
0.2771. Let's check this. So the zero index and the first index is chosen from the first row. From the second row, the
13:44
second index and the zeroth index is chosen. which means this index and this
13:49
index is chosen because these are the largest values. So from every row the
13:54
two largest values are chosen and the indexes of the largest values are given in top k indexes and the indexes of and
14:02
the actual largest values are given in this top k logics right
14:08
u so now what I've done here is that from every row I have essentially selected the largest two
14:14
elements which which means that I have selected which two experts I'm going to
14:19
be routing every token towards. Next thing what I have to do is that I want to make sure that the uh these uh um
14:28
these values for the two experts which are selected they sum up to one essentially. Uh so to do that as we have
14:35
shown over here what I do is I first replace the values which are not selected with negative infinity and then
14:41
I will apply soft max that will ensure that in every row uh we sum up to
14:48
one. So we have to use negative infinity and apply softmax. The way this is done is that uh in these
14:57
two in these two pieces of code what we are doing is that the elements which are not really selected. So if we run this
15:04
right now the elements which are not really selected those are replaced with negative infinity. That's the first
15:09
step. And in the second step what we are doing is that the elements which have been replaced with negative infinity.
15:16
So we are applying soft max at this step. So what that would mean is that wherever negative infinity is there I
15:23
now put all those values to zero. So these are experts which are not selected for that given token. And if you look at
15:29
every row now the elements sum up to one. So the first row first token is routed to expert number one and expert
15:36
number two. They sum up to one. The second token is routed to expert number one and expert number three. They sum up
15:42
to one. Third token is routed to expert number one and expert number three. They sum up to one. Fourth token is routed to
15:49
expert number one and expert number two. They will sum up to one. So until this
15:54
point we have reached a stage where uh we have got this expert selector weight matrix. This expert selector weight
16:01
matrix gives me an indication of which two experts are selected for every token and it also gives me an indication of
16:08
how much weightage I need to give for each expert. So for example, if you are now looking at token number one, I have
16:15
to give 57% weightage to the first expert and 43% weightage to the second
16:20
expert. If I'm looking at token number four, I have to give 55% weightage to the first expert, 45% weightage to the
16:27
second expert, etc. U so this is step number four. Now what we are going to do in step number
16:34
five is essentially whatever we have implemented so far in uh step number
16:39
two, step number three and step number four. We are just going to put it all together in a class called top k
16:46
routing. So this top k router class what it does is that it first
16:51
uh uh it first multiplies the input to the mixture of experts module with the
16:57
router that gives me the logits matrix. Then I do logits top K. Uh so in every
17:03
row it selects the top K experts. Then I replace though the experts which are not
17:08
selected with negative infinity. And then I will apply soft max. So after this forward method in the top K router
17:15
what I get is my expert selector weight matrix. So now we can test this out. Actually if you put the number of
17:22
experts to be equal to three, the top k to be equal to two, the embedding dimension to be equal to 8 and if I
17:30
assume the input to the mixture of experts to essentially
17:35
have one batch of four tokens and embedding dimension equal to 8. You can
17:40
run this. Uh so it says stop k router is not defined. I guess I did not run this
17:46
part. So you have to first run this part of the code and then run this. Yeah. And
17:51
then you'll see that immediately everything works right. So this gating output that's the expert selector weight
17:57
matrix which I have. You'll clearly see that in every row only two experts are selected and both of these experts sum
18:03
up to one. Which means that this class which we have defined now works. So in just
18:10
this small pieces of code I'm able to take the input to the mixture of experts
18:15
module and I'm going to get the expert selector weight matrix.
18:20
Okay. Uh okay. Now what we are going to do is that we are going to come to a
18:26
next step which is something which is called as noisy top routing. So let me
18:31
uh explain about that a bit. All right. So this noisy top K getting is something
18:36
which we had not seen on the whiteboard initially but the main idea is that once we get the expert selector matrix and
18:44
let's say once we have these values we add random gshian
18:51
noise we add random gshian noise to these set of values and the rest of the
18:56
operations remain the same. We do the same top k balancing we do the same negative infinity and the soft max. So
19:03
that portion remains the same. The only thing which changes is that we add some
19:08
amount of noise and the reason it's done is again for load balancing. So earlier we had seen some tricks for load
19:15
balancing, right? Such as auxiliary loss. Then we had seen
19:20
uh the multiplication of FI multiplied by PI. Then we had seen how deepsek
19:25
added the bias term. So adding the noise is essentially a way to make sure that all
19:32
the tokens are not sent to the same set of favored experts. Is essentially introducing some sort of chaos in the
19:38
system so that one expert is not favored. So the idea is that if we add
19:43
noise then in some cases one expert might be more favored in other cases
19:48
some other expert might be favored. So overall addition of noise will balance things out. Uh and that's the reason noise is
19:55
added. So if you take a look at this schematic over here until now what we have done is that we had this expert
20:01
selector matrix then we did load balancing then we applied softmax and then we got the expert selector weight
20:07
matrix when you add noise the only thing which changes is that to the expert
20:12
selector matrix we just add gshian noise that's it and then the rest of the processes remain the same. So now we
20:20
have we are going to create a new class which is called as noisy top router and all the steps remain the same. The only
20:26
difference is that we add this noise logits to the original logits. That's it. And remember what I'm calling logits
20:33
in the code is just the uh expert selector matrix. So we we are
20:38
going to add noise to this. Right? So that's the noisy top router. So you can
20:44
run this piece of code right now. And then what you can do here you can test out with if if I say number of experts
20:51
equal to three top k equal to two embedding dimension is equal to 8. You can run this and then you get the uh
20:58
expert selector weight matrix. So the noisy top router works the same way as the top router. Here also every row only
21:06
has two experts which are active. They sum up to one. So this class is also working. Great.
21:13
Now we come to the next step in the mixture of experts module and that step is essentially now we have to combine
21:19
all the things together. So uh we have got this expert selector weight matrix
21:26
and we have the expert outputs right these are the neural networks which are corresponding to every expert. Now I
21:33
want to somehow merge these two things together. So the way this is done is that you first concentrate on the first
21:39
token and you see which experts it's routed to. So it's routed to expert two and expert three with weightage of 6 and
21:46
0 4. So then you take the expert two and expert three right you take the first
21:53
row of expert two you take the first row of expert three you multiply the first row with 6 you multiply the second row
22:01
with 04 you add all of these together and then you get the output for
22:07
the first token similarly you get the output for the second token uh third token and fourth token so for every
22:14
token which you're looking at you look at the experts which are selected, you look at the weights corresponding to
22:20
that expert and then you take a weighted sum of the corresponding expert output. That's all right. Uh so in this next
22:29
piece of code in the next step we are calling it creating the spar sparse
22:34
mixture of experts module. So here is where we'll actually get the output from the mixture of experts module. Right? So
22:41
what's done in this step is that every input vector or every token is eight dimension and the output is eight
22:47
dimension again. So this this entire process is done for all of the tokens and then we get the resultant output
22:54
matrix which is of the same dimension as the input matrix. So this module which we are going to construct right now it's
23:01
the sparse mixture of experts module which is actually going to give us the output from the input. And for this
23:08
module or for this step to work, we needed our expert selector weight matrix. Right? Without the expert
23:14
selector weight matrix, we cannot go ahead with this module. Without having this matrix, we could not go ahead with
23:20
this module. And that's why it was important for us to first define the noisy top outer which ultimately gives
23:26
us the expert selector weight matrix. So the primary aspect of this
23:32
step involves the expert selector weight matrix. And after acquiring the expert selector weight matrix, the top k values
23:38
in every row are selectively multiplied with the outputs from the corresponding top k experts. That's it. Essentially,
23:45
what it means is that the top k values from every row are multiplied with the corresponding top k
23:52
experts. And this logic which I mentioned to you is the logic which is implemented in this piece of code. So
23:58
this is class sparse mixture of experts. And I'm not going to go through this entire code block in detail because
24:05
ultimately this code block what it does is that it just implements this logic this figure which I have shown you uh in
24:12
the code file. The only difference is that this code block iterates over the
24:17
experts right it does not iterate over the number of tokens. So the way I explained to you right now is I took one
24:24
token I passed it uh I told you that we select the experts for this token. We
24:30
then select the neural networks corresponding to those experts and we take a weighted sum in the code. Uh what
24:38
is done is that we actually loop over the experts. So first we look at the first we look at the first expert
24:45
and we see which tokens are routed to the first expert. Right? If token number
24:50
two and token number four are routed to the first expert, we first get the uh
24:56
output of token number two and token number three. So then we process expert one. Then we process expert two. Now
25:03
expert two has token number one and token number three. Then finally we process expert three that has token one,
25:09
token two, token three, token four. So the result of each token is subsequently
25:14
added as we process different experts. So for example, if expert one has token number two, the output of this is added
25:22
when token number two is again processed from expert number three. So just keep in mind that in the code there is a loop
25:29
over experts because this helps computations that way we can process each expert in parallel. Uh but the
25:37
resulting idea is the same ultimately we are getting the waiting factor the waiting factor or the weight factor for
25:45
every token or we select the top K experts from for every token. We select the
25:51
corresponding top K neural networks and then we take away dead sum. It just can be done in two ways. Either we can
25:57
process every token or we can loop through every expert. It turns out that
26:02
looping through every expert in the code is much more uh computationally
26:08
tractable. Actually I had also written a code which loops through tokens so that
26:13
I can easily explain to you. Explanation is much more easier if I show through every token. But if you write the code
26:20
like that, it becomes very uh slow. So that's why it's it's much
26:26
better to loop over the expert experts instead. And that also makes the code much
26:31
faster. So you can run this sparse MOE class and now you can test this out
26:37
right. So let's see the number of experts which we are choosing to be three. Top K is equal to two. The
26:42
embedding dimension is 8. Dropout I'm assuming to be equal to 0.1 over here.
26:47
And you can run this. So then you get the output. So number of batch let me assume
26:53
one. So then uh actually this should be 1 comma 4, n embed because the number of
27:00
tokens are four. So you see the output is the same as the input. So what we
27:05
have done over here is that this is the input to the mixture of experts module and the dimension is 1a 4a 8. This is
27:11
the same as what we have been seeing over here 1 4a 8. And then we pass the
27:17
this input through the sparse mixture of experts class. Correct? And the sparse
27:22
mixture of experts class needs three things. The embedding dimension, number of experts and top key which we have
27:27
already defined over here. And then we get the output to be the same shape as the input. So this is the output of the
27:35
mixture of experts module. Right? So until now till step number seven essentially what we have coded out is
27:41
that uh if you take a look at this uh whole architecture which we are going
27:47
to code today till step number seven we have actually coded out uh this mixture
27:53
of experts module. So when we get an input to the mixture of experts module we get an output which is of the same
27:58
dimension. So until now we have coded this part which is probably the most important piece of today's lecture
28:05
because this is the main thing which I wanted to show you. Uh now what we are going to do from step eight onwards is
28:12
we are going to put everything together. So first I'm going to put all the building blocks of the mixture of
28:17
experts together and then I'm going to assemble the rest of the language model
28:22
architecture. What were the building blocks which we saw? First we saw this class for an expert and what did this
28:29
class do? This class essentially created a neural network for every expert with this expansion expansion contraction
28:36
architecture. U so this was the first class class expert. Then the second
28:43
class which we saw was class noisy top K router. The whole goal of this class was to give us the expert selector weight
28:50
matrix because without this weight matrix, it's impossible for us to get the output of the mixture of experts. If
28:56
you look at this schematic, this expert selector weight matrix gives us which experts are selected for every token and
29:02
what's the weightage for each expert. This noisy top router class helps us achieve that. And the final class which
29:10
we defined is the sparse or the sparse mixture of experts class. What this class does is that it just takes uh um
29:18
what this class does is the top k values from every row. The
29:24
top k values from every row are multiplied with the top k experts. A weighted summation is done based on
29:30
these values given in the expert selector matrix and then we get the output. The output is of the same shape
29:37
as the input. Right? So you can run this is it's essentially
29:42
putting together all the building blocks of the mixture of experts module. And now what we are going to do is we are
29:49
going to assemble the rest of the architecture. The first thing which we are going to assemble is that
29:55
um we are going to start constructing the transformer block now and the first thing which we are going to assemble is
30:02
the multi head attention module. So and then we'll assemble the rest of the transformer block. So we have created a
30:09
class for head here which is one attention head and I'm not going to go through these details because that's the
30:15
that's not the main purpose of today's lecture but essentially what we do here is that we get the keys we get the
30:21
queries we multiply the uh queries with the keys transpose and uh we also divide
30:29
by the square root of the keys dimension then we implement causal attention and uh finally what we do is
30:36
that the we we get the attention scores, we get the attention weights matrix and we multiply the values matrix with the
30:43
attention weight matrix and that gives us the context vector matrix. That's the main output of the attention mechanism.
30:51
And then in this multi head attention class, what we do is that we just merge the different heads together. So we
30:57
create multiple such heads. That's the main purpose of multi head attention. If you are not clear about this part, I
31:03
have conducted several lectures on self attention, multi head attention etc in
31:08
this series itself. So we have actually gone through all of these sequential steps. How to get the queries multiplied
31:15
with the keys transpose? Why do we divide with the square root of keys dimension? How do we implement the
31:20
causality over here? Then how do we get the uh context? How do we get the
31:26
context vector matrix ultimately uh from the attention weights etc. Right? So
31:32
this is the main purpose of the attention mechanism and then we extended to the multi head attention by just
31:37
aggregating the output of multiple heads. So here is how we have assembled
31:42
the multi head attention block and then we are now ready to code the entire transformer block. So the way we do it
31:49
is that first we have a layer normalization layer. So which is the first uh layer
31:56
normalization after that we have self attention SA. So this SA is the multi
32:01
head attention. So layer normalization is followed with multi head attention and then this X equal to X plus this is
32:08
the shortcut connection. So we have layer normalization multihead dropout shortcut connection. This first part is
32:14
done. And then we again have layer normalization. Then after this second
32:20
layer normalization we have this SM OE. And remember SM OE is the sparse mixture
32:25
of experts class which we have defined earlier. So this SM OE is the replacement of the feed forward neural
32:32
network. That is the sparse MOE class. Remember the sparse MOE class is something which we have defined over
32:38
here. So uh here after the second layer
32:43
normalization we have the sparse mixture of experts. Uh so here after the second
32:48
layer normalization we have the sparse mixture of experts. Uh and then after that we have the shortcut connection
32:57
again. So the dropout which you see here the dropout which you see here and here is already embedded in the sparse class
33:05
and the multiad attention class. So that's why we are not defining it again here. So there are two shortcut
33:10
connection modules. So this is the first shortcut connection. The second shortcut connection. The second shortcut
33:16
connection module involves the sparse uh mixture of experts. So it's this block
33:22
which we have coded earlier. That's the sparse mixture of experts. So until now you can see that
33:27
we have assembled the entire transformer block. So that that finishes the step number 10. And now what we do is that we
33:35
have to now code the entire language model architecture. So let's see how to do that. U all right. So in the next
33:43
step what we are going to do is that after assembling the transformer
33:48
block we have to assemble the input layer and we have to assemble the output layer also only. Then the entire
33:54
language model architecture is assembled. So in the input layer we have to do the tokenization, the token
34:00
embeddings, add the positional embeddings and in the output layer we have to add this logits matrix. So let's
34:06
see how that is done. First we have the token embedding. So the we have the
34:11
token embedding table. We have the position embedding. The token embedding is then added to the position embedding. That's
34:18
the uh that's the input block. That's the input block. The token embedding
34:24
plus the positional embedding. That's the input embedding which essentially then goes to the transformer block. And
34:29
remember we have multiple such transformer blocks. Right? So this class which we have defined is for one
34:35
transformer block. In a language model there are multiple transformer blocks which are chained together. So first I
34:42
have defined this self.blocks blocks where I've chained these multiple transformer blocks together and uh the
34:48
token embedding plus the positional embedding that's the input embedding right that's then passed through through this uh uh that's passed through this
34:56
chain of transformer blocks and then the output which I get that is passed through this uh output layer so this
35:03
output layer has two things we have a layer normalization and finally I have this logits layer so if you look at this
35:10
entire output layer Okay. Um, if you look at this entire
35:15
output layer, it has two things, right? It has a layer normalization part and it has the output layer. This output layer
35:21
takes me from my embedding dimension to my vocabulary size for the next token prediction
35:27
task. So essentially in these steps we have assembled the input, the processor and the output. These three lines of
35:34
code are the input part. Token embedding plus the positional embedding gives me the input embedding. This one line of
35:40
code here is the entire transformer block. So we have assembled multiple such transformer blocks together. Within
35:46
each transformer block, we have uh so many things which are going on within each transformer block. We have the
35:53
mixture of experts module. We have the multi attention module. So it's this line of code where essentially all the
35:59
magic is happening. All right. Uh and then these two lines of code are for the
36:05
output. Right? So here what we are doing is that this here we are doing the layer
36:11
normalization and then we are going from the embedding dimension to my vocabulary
36:16
uh space so that we can do the next token prediction task. Um and then this
36:22
part of the code is just getting us the loss between the next token which we
36:27
have predicted and the actual next token. And then this is the generate token which we can use during evaluation
36:34
and we can also use this the generate function which we can use during evaluation if needed and also during
36:40
inference. So you can run this piece of code and up till now we have finished step number 11 which is defining the
36:46
entire language model architecture. So while we have reached step number 11, we
36:51
have actually finished this entire thing. We have coded out this entire architecture.
36:58
uh but until now I have not shown you the data right uh first we have to so I
37:04
told you about the token embeddings and the positional embeddings but I did not show you the part where the data is
37:09
tokenized I'll I I'll take you through that and I'll also show you uh how the input and the output batches are created
37:17
so I showed you this loss function over here um I showed you this loss function
37:23
um over here right between the logics and the targets uh but I have not yet shown you how the
37:30
targets are created. So let me show you that also. First let me show you how the
37:35
training and testing data is created and how it's tokenized. So we have this input.txt file, right? Which has all of
37:41
the Shakespearean data. So we take every we take every single sentence here and it's converted
37:49
into a bunch of characters. Those are our individual tokens which which are passed to the language model. So here we
37:55
are doing character level tokenization. So uh this what this does
38:02
is that this line just takes text and converts it into individual characters. So we have an encoding scheme which
38:08
essentially maps u maps a string to integers. This is the
38:15
encoding scheme and then this is the decoding scheme which maps the integer back to the uh character. So first you
38:21
take the entire data set and you convert it into characters. Right? But computers can't deal with characters. They need
38:27
numbers. So to get these numbers, every character is essentially encoded as a number. Uh so we create a mapping from
38:35
characters to integers. That's the string to integer and then a reverse mapping which will help us during the decoding stage while doing inference.
38:42
uh so essentially imagine this entire paragraph is this entire data uh which I
38:49
have highlighted right now let's say this entire data is first broken down into characters every character is
38:55
assigned a number so we have a huge set of numbers at this moment uh then we
39:01
create convert this data into training data which is 90% of these numbers and validation data then what we do is that
39:08
we create batches we create the input batches and we create the output batches is the output batch is just the input
39:14
shifted to the right hand side by one. Uh why do we do this? Because that because ultimately we just want to
39:20
predict the next occurren. So this x is my input batch or the input data and y is the output data. I'm going
39:27
significantly faster through this part here because we have covered this a lot in many of the previous lectures and
39:34
here I just want to give you an overall idea for how the input and targets are created. Right? So my input and targets
39:40
are created and the estimate loss function is the loss between the input and the target. Uh not the input and the target
39:47
the it's the loss between the predicted next token and the actual next token for
39:53
each batch. Uh so that's the estimated loss which is the step number 13 define LLM
40:00
loss. And now we move to step number 14 and 15 which is defining the training
40:06
parameters, the training loop parameters, the hyperparameters and then initializing the entire model. So here
40:12
are the parameters which we have defined. Okay. Batch size is 16 which is how many
40:18
independent sequences will be processed in parallel. The block size is the context size which we have set to be 32.
40:25
Learning rate is 10 tous 3. The number of evaluation iterations 400 which means
40:31
after every 400 iterations we are going to print out something. N head that's the number of attention heads is eight.
40:37
N layers is the number of transformer blocks that's eight. Number of experts we are going to choose it equal to
40:43
eight. And top k which means out of those eight two experts are going to be selected for every token.
40:51
Now for the sake of simplicity, if you are on T4 GPU, you can just set this to 20 maximum iterations so that it will be
40:58
easier to run on your local machine. But if you actually want to get decent results, this should be set to as high
41:03
as 50,000. Okay, if you are an if you are on an A100 GPU or if you actually
41:09
want to see good quality results, the maximum iterations need to be very high. I'm just doing a toy demonstration here.
41:16
So I'm setting the maximum iterations to be equal to 20. Uh so here is the weight initialization which we are using
41:23
timing. You can use different initializations also and then we we define uh so I've got an error here. WA
41:31
cap size is not defined. All right. So I'll check where the vocap size is
41:37
actually there. Uh I think I did not define this part. Yeah. So then let me
41:43
try to run this once more.
41:49
name block is not defined. I think I I also did not run the
41:55
uh transformer block code. So yeah, this code I did not run and this code I did
42:01
not run. So that's why I might be getting an error over there. So now I have run all the different code blocks.
42:08
So hopefully this should work. Yeah, now I can see that the entire language model is initialized and the initializations
42:15
are also applied. And now we are ready to run the pre-training loop. Here we are what we
42:21
are doing is that we are doing the forward pass. Uh then we are doing the back propagation using the and then we
42:28
are updating the weights using the AdamW optimizer. You can feel free to play around with these hyperparameters which
42:34
we have defined over here. This is just the beginning and you can use this as a
42:39
reference to try out various different combinations. And then what we do is
42:44
that we uh we loop through the entire data set a huge number of times and
42:50
every time we have a batch we calculate the loss uh then we optimize the uh or
42:57
we update the parameters we do the back propagation right so you can run this now you'll see that this entire
43:03
architecture has around 10 million parameters so it's really a very small language or a tiny language model really
43:09
u and I had run this initially and then after that I had also done the inference part in the last piece of code which is
43:15
step 17 and I had got these results. This was only this was for 200 iterations. So of course the results
43:22
were not very good but I can clearly see that they are following uh what we had in the data right. So the data is in
43:29
dialogue format and just within 200 iterations I am slowly starting to get
43:34
this dialog format over here. I'm pretty sure that if you increase the number of iterations to maybe
43:42
uh 50,000 or 60,000, you'll start getting results which kind of look like Shakespeare in data
43:48
set. Already just within 200 iterations, I was pretty surprised that the model
43:53
learned where to end sentences, how long generally should sentences be and the semicolon style of dialogue which I had
44:01
not expected the model to learn. So right now you can see that my training loop is happening live and I'm only
44:07
doing 20 steps. So this will hopefully finish uh in a faster manner. But the
44:15
whole idea is that within 20 steps you will never get good results. You should run it for a huge number of steps only
44:22
after which you'll start seeing a decent reasonable results.
44:27
So uh what we have done in this code file is that we have coded an entire mixture of experts module from scratch
44:33
and we have done it in just 17 steps. This is incredible right now. Whatever
44:39
my whole goal through these lectures is that I want to show you things on the whiteboard as we have been seeing but I
44:46
also want to help you to code things from scratch. So now we can see that the training loss has become this much and
44:52
we can do quick inference. This is just 20 iterations. So I don't expect to see good results but still uh so my whole
45:01
aim through these lectures is so that you become good at theory and fundamentals. So you should visualize
45:07
matrix multiplications very easily when matrix multiplications come in front of you but at the same time all of you
45:14
should develop an intuition about the code and you should you should be very good at coding things from scratch and
45:21
building things from the ground up. Why am I showing these uh Google collab code
45:26
files which show you how to build things from ground up? The reason I'm showing these code files is because if you want
45:32
to do research and if you want to build upon something, if you want to build the
45:37
next mixture of experts innovation, how will you do this? The only way you will be able to do this is if you are
45:43
comfortable with the theory, you know how the theory works and you know the nuts and bolts of how the entire code is
45:50
assembled. So as an exercise what you can do is that you can take this code and you can
45:55
also uh uh maybe you can add expert capacity to this. So if you remember in
46:03
the previous lectures we have learned about auxiliary loss. We have learned about load balancing and we have also
46:09
learned about capacity factor. Right? In the code which I have shown to you capacity factor is not yet added. Maybe
46:16
you can add that. Then what you can do is that we have seen deepseek innovation right the first deepseek innovation is
46:22
this auxiliary loss free load balancing where they added this bias term uh I've
46:28
completely explained to you how this bias term works and it's a very simple
46:33
addition maybe you can take the code which I've just shown you and modify it with the addition of this bias term and
46:39
then you can do some experimentations with it you can play around with different uh settings and hopefully or
46:45
maybe that leads to a new research direction for you. Uh the third thing
46:51
maybe you can do is that uh they have the shared experts right. So Deepseek had this architecture where they had
46:58
shared experts uh which was a common set of experts and then they had routed experts as well. So
47:06
I have not shown that in the code right now. Maybe you can do that. Maybe you
47:11
can do these modifications. Right? So you can use this code as a starting point to do wide range of
47:17
experimentations. You see this is the output which I have got from 20 iterations. You can already see that the
47:22
output for 200 iterations was much better than this because here that colon structure is not identified by the
47:29
model. Uh but now the sky is the limit for you. Take this code run it for a huge number of iterations. do the
47:36
modifications as I mentioned with maybe shared experts maybe with the bias term
47:41
and hopefully that sets a new research direction for you. Thanks everyone. I hope you enjoyed these five lectures on
47:48
mixture of experts. We will now move towards more innovations in the deepseek series. Lots of exciting stuff is yet to
47:55
come such as multi-token prediction etc. So we'll see about that in the next lectures. Thanks everyone and I look
48:01
forward to seeing you in the next lecture.






