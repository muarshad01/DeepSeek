
0:05
Hello everyone and welcome to this lecture in the build deepseek from scratch series. Today we are going to
0:12
start understanding the quantization methods which deepseek actually implemented in their version 3 report.
0:19
So if you look at the deepseek version 3 technical report PDF they have a section
0:25
called FPA training or floating point a training and in this section they
0:30
implemented several aspects with respect to quantization and I've divided those into five major aspects mixed precision
0:38
framework fine grained quantization increasing accumulation precision
0:44
mantisa over exponents and then online quantization. In today's lecture, we'll
0:49
cover mixed precision framework and fine grain quantization. In the previous lecture,
0:55
we have had a introductory look at quantization and some of these concepts
1:00
will be relevant to you in today's lecture as well. Essentially, the main idea of quantization is that
1:07
parameters take up space. Every parameter in a large language model takes up space.
1:13
So if the parameter is represented as a floating point 32bit number, it takes up
1:18
more space than let's say if it's represented as floating point 16 etc. So
1:24
the main aim of quantization is to reduce the precision of a model's parameter from higher a bit let's say
1:30
from 32bit to a lower bit let's say 8 bit or 16 bit. So through this the main advantage
1:37
which we get is that we reduce the uh total memory requirements but at the
1:44
same time we don't pay that much of a heavy price with respect to the accuracy of the model. Of course, the accuracy
1:51
slightly reduces because now every parameter will be represented in a low precision. But that much reduction in
1:58
accuracy of the large language model is typically fine for us because it does not degrade the performance that
2:04
much. In today's lecture, you'll come across floating point 32 which consists
2:09
of 32 bits. Then you'll come across floating floating point 16 which consists of 16 bits. floating point 16
2:17
the range is much reduced compared to floating point 32 you'll also come across BF16 so the range of BF16 is same
2:25
as floating point 32 but its bits are the same as floating 16 so it still has
2:31
16 bits b16 is brain float 16 then there
2:36
is also integer 8 whose range is very small minus 127 to 127 and it has 8
2:42
bits one more concept which we'll see when we come to fine grain quantization is that when you want to actually
2:49
quantize and go from a floating point 32 let's say to a floating to an integer 8
2:55
what you do is that if you have a sequence of numbers in floating point 32 you divide all these numbers with the
3:01
highest number in this case 10.8 and then what you will do is that you will multiply it with
3:07
127. So keep this in mind that we divide by the highest number and then multiply
3:12
it with 127 if we want to let's say quantise from a floating point 32 to an integer
3:18
8. All these concepts which I'm showing you right now we have covered it in previous lecture. But uh this quick
3:25
summary will be enough for you as we get started with today's lecture. So let's
3:30
get started with the first major concept which deepseek implemented and that's
3:35
something called as the mix precision framework. So if you look at the deepse paper they have the mixed precision
3:41
framework in section 3.3.1 and they have a schematic which is corresponding to this which looks like
3:48
uh here. Now based on the introduction which I gave you you'll already understand FP8 BF16. So this floating
3:56
point 8 a brain float 16 floating point 32 uh BF16 etc. But now let me explain
4:04
to you what this frop means, what this drrad means, what wg grad means
4:10
etc. So when you look at a large language model there are number of parameters which get multiplied
4:17
together. The typical operation which happens in a simple neural network layer is as follows. We have y = w * x where x
4:26
is the input, w are the weights and y is the output. So a forward pass or a
4:32
forward propagation which is referred to as f prop over here are the operations in which weights are multiplied by
4:39
inputs. So y = w into x and now I'm going to explain to you how deepsek
4:47
stored the weights stored the inputs and the output in which quantized format.
4:53
Okay. So when the in so let's first take a look at the inputs when the inputs
4:58
were used in this formula they were converted from a brain float 16 to a
5:04
floating point.8 8. Okay. U so floating.8 as you must
5:10
have seen it has a much lower range compared to brain float 16 and of course much lower number of bits. So it takes
5:16
less memory. Weights on the other hand are maintained in a high precision which means they are stored in brain float 16
5:24
or floating point 32 but the weights are converted into a floating point.8 on the fly. So when you take this
5:30
multiplication you have a floating.8 8 weight multiplied by floating.8 input
5:36
which gives me the output and the output is computed initially in floating point
5:41
32 but for numerical stability it's converted back into brain float 16. So
5:46
whenever we have forward propagation modules anywhere in the large language model what deepse did was that they did
5:53
this multiplication of a floating point 8 and a floating point 8. So it took up less memory and the output they computed
6:00
the output initially in floating point 32. Remember that when a floating point 8 is multiplied by 14 floating.8 we get
6:08
a higher decimal representation. So we can we can get the output in a higher
6:13
format compared to the inputs. So the output is computed initially in floating point 32 for numerical stability but
6:20
then it's converted back and stored in BF16 to optimize the memory. Remember
6:25
BF16 has 16 bits and it covers the entire range as floating point 32 which
6:30
we already saw over here. If you remember this was the uh BF16 which we saw. It covers the
6:37
same range as floating point 32 but it has 16 bits. So it takes up less memory compared to floating point 32. So now
6:45
you'll be able to understand this f-prop block over here. Right? If you see the f-prop block has inputs which are
6:51
floating 8 weightights which are floating 8 the inputs and the weights are multiplied and the output is
6:58
initially computed as floating point 32 but it's stored in memory as bf16 or
7:03
brain float 16 that's the way to interpret this f-prop
7:09
block along with forward propagation we also have backward propagation so if you
7:14
look at a neural network like this and these are the inputs of a particular real layer and these are the output.
7:20
Remember we are doing backward propagation right? So when we reach a particular layer we'll have the partial
7:25
derivative of the loss with respect to the output of that layer and then we'll need to find two things. We'll need to
7:31
find partial derivative of loss with respect to weights because we'll need to update the weights of this layer and
7:37
we'll also need to find the partial derivative of loss with respect to the inputs of the layer. Why do we need to
7:43
find the partial derivative of loss with respect to inputs dl by dx? because this dl by dx will serve as the dl by dz of
7:51
the previous layer. So when you look at a given layer, you'll need to find the partial derivative of the loss with
7:57
respect to the weights and the partial derivative of the loss with respect to the inputs. The partial derivative of loss
8:04
with respect to inputs is given by this formula dldx which is also called as dgrad over here. So this d grad is the
8:11
partial derivative of loss with respect to the inputs and it's given by the formula dl dz multiplied by the weights
8:19
transpose. Now to get the partial derivative of these inputs the weights are again converted on the fly to
8:25
floating point.8 8 for this calculation and dl by dz is stored as braille float
8:31
16 but it's converted to floating point8 for computation and dl by dx is computed
8:36
initially as floating point 32 then converted to brain float 16 so if you take a look at this entire computation
8:43
dl by dz is on the fly converted to floating 8 weightights are on the fly
8:49
converted to floating point8 and the multiplication it's computed as floating point 32 and stored as bf16 or brain
8:56
float 16. So this is exactly what you see over here, right? Weights are floating 8. Weights are multiplied with
9:02
the output gradient which is floating.8. The multiplication is initially computed as floating point 32 but it's stored as
9:10
brain float 16. That's my input gradient. But I also need the gradient with respect to the weights. Right?
9:16
That's W grad. Uh and W grad is given by this formula. partial or the gradient with
9:22
respect to the weights is equal to xrpose multiplied by dldz and x is
9:28
stored as floating point 8 dlz again is converted from brain float 16 to floating point8 and keep in mind that dl
9:37
by the partial derivative with respect to the weights is stored as floating point 32 to ensure numerical stability
9:43
for weight updates. So this is not converted into floating point 8 because this will ultimately lead to my weight
9:51
updates right. So if you look at dl by dw you'll see that the weight gradient right which is computed after
9:58
multiplication of the inputs inputs multiplied with the output gradient the weight gradient is computed
10:04
as floating point 32 and it's stored also as floating point 32 it's not stored as floating 8 or bf16 it's stored
10:12
in a very higher precision and higher number of bits um so this is stored as floating point
10:19
32 as you can see over here and then we have these weight updates. So this is the optimization
10:26
state and the for the optimization the weights the initial weights are stored as floating point 32. DL by DW of course
10:33
is stored as floating point 32. So the initial weights minus this DL
10:39
by DW that gives me my updated weights and that's stored as floating point 32. Right?
10:45
So after the weights are updated the master weights are converted to BF16 for next iteration forward pass or they are
10:52
converted into floating point 8 when needed. So these weights when they are utilized for calculations they are
10:58
converted on the fly to floating point 8. But remember that the master weights are computed as floating point 32 and
11:04
the partial gradient with respect to the weights is computed as floating point 32. So this portion is shown in this
11:11
part of the schematic over here. We get the weight gradient in floating point 32 and we get the master weights as
11:17
floating point 32 and they are on the fly converted to floating point 8 to get
11:22
my u forward pass to get my degrad etc. Okay. So this is how
11:29
the different parameters are stored in different quantized formats when we are
11:35
doing the forward pass and when we are doing the backward pass. Okay. So now I
11:40
hope that when you look at this figure you'll actually completely understand what's going on. Figure six. If you
11:46
directly look at this figure it's very difficult to understand what's going on. But now when you look at it you'll
11:52
understand what the mixed precision framework actually means. Uh and so now if you compare the
12:00
advantages of what exactly we are doing here. Why do we do this mixed precision? Right? So you see we are converting
12:08
several variables into floating point 8 on the go. This significantly reduces the computational and memory cost and it
12:15
provides us a big speed improvement uh especially compared to using BF16. So if
12:21
you look at the schematic now there are several variables which are converted to floating point 8 on the fly for
12:27
calculations during the forward pass as well as during the backward pass.
12:33
This speeds up the computations by a lot. And GM means matrix multiplication
12:38
operations. General matrix multiplication operations. They involve a lot of multiplications like this. So
12:46
using a floating point significantly speeds up my computations and it reduces my memory cost. But you see we are
12:54
smartly uh also retaining some states in higher precision, right? master weights
13:00
and the optimizer states are stored in floating point 32 to ensure training stability. That's why this is called as
13:06
mixed precision. During computations, some variables which are not as important are converted to floating
13:13
point on the fly. But variable states which are important so for example these master weights for example then partial
13:20
derivative of loss with respect to w those are maintained as floating point 32. That's why it's called as mixed
13:26
precision. uh now in the paper deepse mentioned that embedding modules output heads
13:33
gating modules normalization and attention operators are sensitive and hence these are retained in higher
13:39
precision that's an important thing to understand so if you take a look at this section which they have mentioned they
13:46
mention here clearly that which operations are retained in higher precision uh for this reason after careful
13:53
investigation we maintain the original precision for the following components the embedding module Token and
13:59
positional embedding. The output head where the final output head converts from a embedding dimension to the
14:05
dimension of vocabulary size. Mixture of experts gating modules, normalization operators and attention operators. The
14:13
floating points or the precisions are higher for these operations either their BF16 or floating point 32. Whereas let's
14:20
say all the other areas where we have weight products, we have weight multiplications, those are calculated
14:26
using the floating point.8 format. So you see we are using low precision but at the same time for some calculations
14:33
we are using high precision. So we get memory reduction, computational efficiency and numerical stability at
14:39
the same time. It's a very smart way for doing operations. Operations which are not important uh or we can we can do
14:47
fine with reduced precision in those operations we convert those into floating point 8. Whereas operations
14:52
which are important uh those are retained in floating point 32 especially in mixture of expert gating modules and
14:59
attention operators embedding modules etc. All of those are retained in higher precision.
15:06
So this is the main concept behind understanding mixed precision framework and that's the first quantization
15:12
technique which deepseek implemented. U along with this the second quantization
15:17
technique which is also shown in their second figure which they implemented is something which is called as fine grain
15:23
quantization. So let's come to understanding this right now. Okay. So if you look at the fine grained
15:29
quantization section in deepseek you'll see that they have these figure 7 and
15:35
figure figure 7 a and figure 7b just looking at these figures it's really
15:40
very difficult to understand what's going on there there are usage of so many colors here um there is something
15:46
called tensor core cuda course gam wgmma scaling factors so many things are going
15:53
on the first thing to understand is that this figure actually consists of two figures. There is fine grain
15:59
quantization and there is increasing accumulation precision. First we'll understand fine
16:05
grain quantization which is the figure on the left hand side. So to understand this the most important uh thing to
16:12
understand is this uh conversion which we saw. So if you want to go from let's say floating point 32 to integer 8. What
16:19
we do is that if you you take the sequence of numbers you divide with the highest number and then what you do is
16:26
that you multiply it with 127 because we're converting it to integer 8. So the important thing to remember is that we
16:33
divide the original number or the original sequence of numbers with the highest absolute
16:39
value and that's where we start fine grain quantization. So for example if you see fine grain
16:47
quantization starts with the scaling solution. So you have the initial sequence of numbers. What you do is that
16:53
you take the maximum value of these numbers and then you multiply it with the uh if you're converting it to
17:00
floating point 8 you multiply it by 127 over here. So if you have the sequence of numbers 2 3 and four and if you want
17:07
to convert it into floating point 8 what you do is that you first uh divide these
17:12
by four because that's the highest number and then you multiply it by 127 and that's how you get the conversion
17:19
into floating.8. The reason this is done is that u when we move to lower precision
17:27
formats like floating point 8 we have fewer exponent bits. Thus the numerical range which we can represent is
17:35
small and that can lead to overflow problems or underflow. So overflow is when the numbers become too large.
17:42
Underflow is when the numbers become too small and we lose the precision. Uh so that's why the scaling solution is
17:48
employed because we can scale the entire tensor based on its largest absolute value and scaling just like
17:54
normalization in deep learning it reduces this underflow or underflow or overflow issues. So this method is very
18:02
frequently used to convert a sequence of numbers into floating
18:07
point 8, floating point 16 etc. Essentially to quantize a sequence of numbers but this approach has a big
18:13
weakness. The major weakness of this approach is that even a single outlier can drastically reduce the represent
18:20
representation accuracy of a whole tensor. And let me explain to you what I mean by this. First if you take the look
18:28
without outlier 2 3 and 4 we divide all these elements with four and then we
18:33
multiply it with 127. So this is my uh this is my
18:38
quantized tensor now 63.5 95.25 25 127 and de quantization becomes much easier
18:45
right then I just multiply it with 4 by 127 for quantization I multiply it with
18:51
127x 4 for deontization I multiply with 4 / 127 and I recover back the same
18:57
tensor which I had before which means I do not lose any precision but now consider the case when we have an
19:04
outlier over here if we have an outlier which is way larger than the other numbers then what do we do We divide
19:11
these numbers with 500 and then we multiply with 127. So the scale tensor
19:16
actually now looks like this. But remember we are quantizing to floating.8 right? So there is limited
19:23
precision. I cannot represent 0508 as a floating point 8. I cannot represent 762
19:29
as a floating.8 1.016 I cannot represent as a floating point 8. So these numbers when
19:36
you actually quantize them they become like this. So 050 it loses its precision and it becomes 0.5 762 loses its
19:44
precision and it becomes 75. 1.016 loses its precision and becomes one. So the
19:51
quantise tensor loses its precision. And when you dequantize back you get numbers which
19:56
are which are different from the original set of numbers. Right? And this is because the quantized version has
20:03
lost lost its precision. um this is the main issue with having
20:09
outliers. So in the current approach which we are doing we take all of the values and then we divide it by the
20:14
largest possible value right that leads to this major issue that smaller values
20:21
can significantly lose precision. So 2 3 and 4 are much smaller compared to 500.
20:27
Right? So when you divide all of these by 500 and then you multiply it with 127
20:33
you cannot represent 0508.762 1.016 as
20:38
floating.8. So you have to represent them as 0.5 751. So they lose their
20:43
precision and when we dequantize back we get numbers which are not the same as the original numbers. That might affect
20:49
the accuracy when I train my large language model. Ideally I want my dequantized numbers to be as same as the
20:56
quantized numbers. So what is the solution for this? The solution for this is something
21:02
which DeepS seek implemented in their paper which is called fine grain quantization. So if you take a look at
21:08
section number 3.3.2 they have this subsection
21:14
called fine grain quantization and the first figure on the left is with respect to the same thing. So what they are
21:21
doing here is that instead of scaling all the numbers by one single number, we break down the
21:28
uh we break down the entire activation outputs into chunks. So let me tell you what this means. Um when you are dealing
21:36
with large language models, we typically have vectors, right? Let's say this is a
21:41
token for the the input embedding token can be a 256 dimensional vector. Earlier
21:47
what was done is that we we took this 256 dimensional vector we looked at all
21:53
the values and then we found the maximum value we found the maximum value and
21:58
then we scaled all these numbers by the maximum value and let's say multiplied it with 128 etc. Right? That's the
22:05
approach which I mentioned over here. But now what fine grain quantization means is that you divide this vector
22:12
into chunks. The first chunk is of 128 values. The second chunk is of 128
22:17
values. So this first chunk so this first chunk or this first group is scaled by the maximum value only in that
22:25
group. So in this group the maximum value is 20. Right? So the first group will be scaled by 20 and the second
22:32
group is scaled by the maximum value only in that group. So if you look at the second group now the maximum value
22:39
is 0.1. So the second group is scaled by 0.1. So we have separate scaling
22:44
factors. That's the most important thing to understand. If you had a common scaling
22:51
factor, if you had a common scaling factor, all of these numbers would be scaled by that factor. Right? So the
22:57
second group would be scaled by 20 unnecessarily. That would cause these smaller numbers to lose their precision.
23:04
But now this second group is not scaled by 20 because 20 is the largest number in the first group. The second group
23:10
does not care about that at all. The second group only looks at the largest number within that group. So it's scaled
23:16
by.1. So we retain the precision in the second group. So the separate scaling retains the
23:22
accuracy and the precision for the second group despite a larger outlier in the first group. That's the main
23:28
advantage which is given by a separate scaling and that's why it's called fine grained quantization. Why fine grained?
23:35
Because we we are dividing the input vector into chunks. Now we also have weight
23:41
matrices, right? Several operations in the uh entire transformer architecture has
23:48
weights such as there are trainable weight matrices for queries, keys, values, etc. So many
23:56
weight matrices are there. So if we have a weight matrix which is 256, 256, we'll
24:01
divide it into chunks of 128x 128. So there will be four chunks of 128, 128.
24:09
And then each of these chunks will have their separate scaling factors. So the first chunk matrix which is W11 will
24:16
have a separate scaling factor. The second chunk W12 will have a separate scaling factor. The third chunk W21 will
24:24
have a separate scaling factor. And the fourth chunk W22 will have a separate scaling factor. So each block W11 W12
24:32
W21 W22 is scaled individually. So if W11 let's say has an outlier it
24:39
won't affect W12 W22 and W21 those won't be those will be unaffected by that
24:45
outlier which is good for us. So if W11 has an outlier only W11
24:52
scaling is affected while W12 W21 and W22 remain precise their precision is
24:57
not lost. So fine grain quantization the name seems difficult but it's actually
25:02
very simple. You take the vector or you take the matrix you break it down into chunks and you have a separate scaling
25:08
factor for each chunk. So now actually we can understand what is going on here. So if you see the this part is actually
25:15
the different scaling factors. So here we have the the inputs and the weights
25:22
right. So remember I told you we have inputs multiplied by weights y = w * x.
25:28
So this x is what is called as an input over here. This is the first case which I told you the activation outputs that
25:35
they are calling as inputs in the figure because this is a vector and the vector will be multiplied by weights which is a
25:41
matrix. So this is the input and this is my weights right. So if this is my input
25:46
vector which is shown by the green color right now. Uh actually let me take a
25:52
screenshot of this and bring it to the whiteboard so that I can explain it to you clearly. So this is my um input
25:59
vector right now which is exactly what I showed over here and what they have
26:04
shown with this different shades of green is the different chunks. So NC I think is 128. We break it down into
26:11
chunks of 128 and then every chunk has a different scaling factor that's shown with this different shades of blue. So
26:17
every chunk has a different scaling factor. That's what's meant by this first part of the figure which I showed
26:23
you over here. Now the second part of the figure which is over here is corresponding to the
26:30
weights. Now this figure I believe could have been a bit better because they could have directly show the matrix over
26:36
here. But what they are trying to show over here is that we have the weight matrices right and those are also broken
26:42
down into separate chunks. So these weight matrices are broken down into chunks of 128x 128. That's what's shown
26:48
over here. It's these chunks and each of these chunks also have a separate scaling factor which is shown by the
26:54
shades of purple color over here. So that's the this second schematic and
26:59
then we also have this tensor core and the CUDA core over
27:04
here. So that I have explained over here. So if you look at the tensor core the green it's a green rectangle
27:11
multiplied by yellow right? So that's actually y is equal to or this is equal
27:16
to w x * uh this is x * w these are the
27:22
activations and these are the weights and we have the output over here. So
27:27
game you when you read the paper you will see terminologies like game. So this is general matrix multiplier these
27:33
computations happen inside Nvidia tensor cores. That's why it's called tensor core over here. It's not a coda
27:40
computation or not a GPU computation. when you want to uh um up there are
27:46
upscaling factors right because until now what we are doing here is that these multiplications happen in lower
27:52
precision as we have seen before the multiplications especially if you look in the forward pass this multiplication
27:58
happens in floating point 8 multiplied by floating point8 so this is lower precision lower precision operations are
28:05
all computed on the tensor core but then once we get the lower precision matrix we are going to upscale it by by
28:12
dequortizing it and by multiplying it with the upscaling factor. So we need to bring it back to a higher precision,
28:17
right? Uh for example, if you take a look at
28:25
um the master weights or these uh if you look at yeah the partial
28:32
derivative of loss with respect to W, it's floating.8 multiplied by floating.8. But the product we need to
28:38
take to higher precision that is done on the QA core. So this second uh this
28:44
second schematic is the output is stored on the CUDA core where there are scaling factors. So the output from the tensor
28:50
course which is floating.8 the the output from the tensor course is floating.8 that is
28:57
scaled back up to floating point 32 for stability. So here element wise scaling
29:02
up or upscaling operation is performed on the CUDA core. That's why there are these uh if you take a look at the
29:08
schematic there is the tensor core over here where the floating point 8 operations are performed and the result
29:13
is also floating.8 and the output is coda core where we do upscaling and store as floating point 32. So this
29:20
entire schematic over here actually has a lot of things which are going on if you if I take a screenshot of this and
29:26
bring it over here and without understanding quantization it's very difficult to understand the schematic.
29:33
So here they are showing the different scaling factors for the inputs or the activations. Here there are different
29:39
scaling factors for the weights. That's the fining rate quantization part. Here we have the tensor core calculations
29:45
which are floating.8. And here we have the output calculations which are on the cod. Those are higher precision floating
29:52
point 32 computations. So there are actually four parts to this one schematic.
29:59
That's why reading this paper is so tough because every single aspect can be broken down into multiple aspects and
30:05
that's why the series right now has become almost uh quite long but to reach
30:10
these 16 pages it's important for us to dive into the very basics if we want to build deepseek from
30:17
scratch. So in this lecture I plan to cover only this much where we have fine
30:22
grain quantization and in the first part of the lecture we covered mixed precision framework. In the next lecture
30:28
I'll cover three other things increasing accumulation precision mantis or
30:33
exponents and online quantization. You'll see that all these three things are come after the fine grain
30:40
quantization. So after fine grain quantization we have increasing accumulation precision. Then we have
30:45
mantisa over exponents and then finally we have online quantization and then the
30:50
uh whole FP8 quantization part of deepseek will be finished. So thanks everyone. We are now
30:57
going deeper and deeper and deeper into deepseek. I don't believe or I believe these
31:02
this thing which I'm teaching right now is not covered anywhere else because we are going into the very basics of
31:08
exactly what these schematics mean. And my goal is to try to explain this to you in as much detail as possible without
31:14
leaving out anything. That's why yesterday's lecture I had on the intro to quantization. And today we covered
31:20
mixed precision framework and f and fine grain quantization. Quantization is
31:26
extremely important when you are developing foundational models and when you want to reduce the memory
31:32
requirements, increasing computational speed etc. So these lectures will be very helpful to you if you are looking
31:38
to build your own foundational model which I highly encourage you to do so if you have reached this part in the
31:43
series. So we have until now we have covered latent attention mixture of experts multi-token prediction and now
31:49
we are almost at the end of covering quantization. Thanks everyone and I look forward to seeing you in the next
31:55
lecture.
