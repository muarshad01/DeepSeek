#### Code Mixture of Experts (MoE) from Scratch in Python
* [makeMoE: Implement a Sparse Mixture of Experts Language Model from Scratch](https://huggingface.co/blog/AviSoori1x/makemoe-from-scratch)
* [minGPT](https://github.com/karpathy/minGPT)

* __Step 0__: Load packages and import data

***

* 5:00

* T4 GPU
* A100 GPU

* __Step 1__: Define each expert as a NN network

* __Activation Function__: ReLu

***

* 10:00

$$Input ~Matrix \times Routing ~Matrix = Expert ~Selector ~Matrix$$

* __Step 2__: Implement the Router

* __Step 3__: Implement top-k load Balancing

* __Step 4__: Use -inf and apply softmax

***

* 15:00

* __Step 5__: Create a class for Top-K Routing

***

* 20:00

* __Step 6__: Create a class for Noisy Top-K Routing

* __Step 7__: Create the Sparse Matrix of Experts (MoE) module


***

* 25:00


* __Step 8__: Putting together all the building blocks of MoE model

***

* 30:00

* __Step 9__: Code the entire Transformer block: Part 1 (Multi-head attention)

***



what's done in this step is that every input vector or every token is eight dimension and the output 
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













