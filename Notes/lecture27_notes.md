
***

* 10:00

***

* 15:00


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



***


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



