sic]
0:05
Hello everyone and welcome to this
0:08
lecture in the build deepseek from
0:10
scratch series. Today is the second
0:12
lecture of exploring deepseek
0:15
quantization. In the previous lecture we
0:17
have looked at how deepseek implemented
0:19
the mixed precision framework and fine
0:21
grained quantization framework. So if
0:24
you have not been through the previous
0:25
lecture you can go through that and in
0:27
this lecture we'll look at how deepsek
0:29
implemented these three aspects. Uh the
0:32
first aspect is increasing accumulation
0:35
precision. Second is mantisa over
0:37
exponents and third is online
0:40
quantization. So to construct these
0:43
lectures I'm following this FPA training
0:46
section which deepse had. So in the
0:48
first lecture on quantization I just
0:50
explained to all of you what is
0:52
quantization how parameters can be
0:54
represented with different bit numbers
0:56
and how we can take let's say floating
0:59
point 32 and quantize it into floating
1:01
point 16 brain float 16 int 8 or
1:04
floating point 8 right and in the second
1:07
lecture which was the last lecture we
1:09
covered the mixed decision framework
1:11
which is this schematic so we covered
1:14
section
1:15
3.3.1 so this is deepseek version three
1:17
PDF which I'm showing right now and we
1:19
also covered fine grain quantization
1:22
which is the left hand side of this
1:23
figure. Today we are going to cover the
1:25
right hand side of this figure which is
1:27
increasing quant accumulation precision
1:30
and uh then we will see mantisa over
1:33
exponents and then we are going to see
1:35
online quantization. So let's get
1:37
started with the first main technique
1:40
which we are going to see today
1:41
increasing accumulation precision. And
1:43
if you see the schematic with respect to
1:46
this again it's very difficult to
1:49
understand what's going on. WGM MA
1:52
then what is G emm then there are
1:56
different colors over here. It's
1:57
impossible to figure out what's going
1:59
on. So my goal in today's lecture will
2:02
be to not dive extremely deep in these
2:04
terminologies but to give you enough of
2:06
an overview so that you are not scared
2:08
of this figure so that you understand
2:10
what this figure actually represents.
2:13
Okay. So, uh in today's lecture, we are
2:17
going to start from this place in the
2:18
lecture notes which is increasing
2:20
accumulation precision. So, the reason
2:22
we are learning about this in the first
2:24
place is that when you perform GIM
2:26
operations, so what is It's
2:28
general matrix multiplication. So, when
2:31
you perform matrix multiplication
2:33
operation,
2:35
uh we have things like Y = W into X + B,
2:38
right? And if this is a floating point 8
2:40
and if this is a floating point 8 uh we
2:43
quickly lose accuracy due to
2:45
intermediate results being too small
2:47
that leads to underflow
2:49
issue. So that means that if two
2:52
floating point eight numbers multiply.
2:54
So if you reduce the accuracy of two
2:56
numbers and if you multiply them and if
2:58
you again perform various such
2:59
operations uh we quickly lose accuracy
3:02
due to intermediate results being too
3:04
small.
3:06
um and this is called as limited
3:07
accumulation precision. So if we do the
3:10
operations on tensor cores as we saw in
3:13
the previous lecture if we do the
3:15
floating point 8 operations on tensor
3:17
core it it has limited accumulation
3:20
precision. So Nvidia tensor coursees
3:22
accumulate game results internally with
3:25
a limited precision far below floating
3:28
point 32 accumulation precision. So if
3:30
you calculate some results and if you
3:32
add them together, if you store the
3:34
results on a tensor core, they they will
3:36
always be of a lower precision around 40
3:39
14 bits. But that's not good because
3:41
that leads to
3:43
uh errors that leads to numerical
3:46
errors. So we would ideally want to save
3:48
these parameters in as high a precision
3:51
as we want on the CUDA core, not on the
3:53
tensor core. So when performing game
3:56
operations with let's say large matrices
3:58
of inner dimensions K low accumulation
4:01
errors low accumulation precision can
4:03
lead to significant numerical errors. So
4:06
for example if you multiply matrices
4:08
with inner dimension K equal to 4096 and
4:12
if we are using low accumulation
4:13
precision that can lead to an error as
4:16
large as 2% which heavily impacts the
4:18
model accuracy. This is not quite good.
4:20
So what we are essentially suggesting is
4:22
that there is a tensor core and there is
4:24
a GPU core right. So there is a tensor
4:26
core and there is a CUDA core. If we
4:29
perform operations on tensor core those
4:31
are floating.8 operations and so the
4:34
resultant accumulated matrices they'll
4:37
be of a lower precision and if we store
4:39
these parameters or these matrices in
4:42
lower precision it can lead to
4:43
significant errors which is not good.
4:46
This is called as limited accumulation
4:48
precision. So what is meant by
4:50
accumulation precision is accumulation
4:52
simply means summation. So if you have
4:54
multiple parameters you'll need to sum
4:56
them up together in memory that is
4:59
called as accumulation. So if this
5:01
accumulation has a lower precision which
5:03
is
5:03
floating.8 or let's say floating or 14
5:07
bits that is much lower than the 32-bit
5:10
floating point precision right so if we
5:12
store these accumulated variables in low
5:15
precision it can lead to significant
5:18
numerical errors which is not quite
5:19
good. So that is why deepseek increased
5:22
deepseek mentioned that we need to
5:25
increase the accumulation precision. So
5:27
how do we increase the accumulation
5:29
precision? So what the authors propose
5:31
is that let's say we get these
5:33
calculations right we get the
5:35
calculations in a tensor core we can
5:37
periodically move them to the CUDA core
5:39
and that is what is shown over here. So
5:41
this uh pink light pink dot is the
5:44
tensor core and it's periodically moved
5:46



***



to the cuda core which is a dark pink
5:48
dot. So what that means is
5:51
that we temporarily move intermediate
5:54
accumulation results from the low
5:56
precision tensor course to high
5:58
precision CUDA course which is floating
5:59
point 32 and we do this periodically
6:02
during computation. This is called as
6:04
promotion to CUDA course. In a very
6:07
simple schematic what we are doing is
6:08
that we get the accumulated results in
6:10
the low precision tensor core and we
6:13
transfer these accumulation results to a
6:15
high precision CUDA core. And this is
6:17
exactly what is uh seen in the
6:19
schematic. So this is a tensor core
6:21
which are low precision calculations. So
6:23
the green multiplied by yellow let's say
6:25
gives me pink. That's a low precision
6:27
value that's periodically transferred to
6:30
a high precision coda code that's shown
6:32
by this dark pink color over
6:34
here. So if I take the screenshot of
6:37
this and bring it on the whiteboard, I
6:40
can show this to you uh in a clearer
6:42
manner.
6:44
So the low precision calculations here
6:47
are transferred to the high precision uh
6:50
high precision cod and this is exactly
6:52
what we are going to see below. So this
6:54
promotion to Kyoda force is done in two
6:56
steps. So the first is low precision
6:59
accumulation MMA. So low precision MMA
7:03
accumulation means low precision matrix
7:05
multiply accumulator. So initially
7:07
matrix multiply accumulate operations
7:09
are performed using floating point
7:11
precision on tensor cores. That's
7:13
essentially this this
7:15
part. Low precision operations are
7:18
performed using floating point 8
7:20
precision on tensure cores. The
7:22
intermediate results accumulate
7:23
internally with a limited precision 14
7:26
bits. So this light purple which we have
7:28
shown here that's a low precision
7:30
accumulation. So the results are
7:32
accumulated in the low precision. It
7:34
might be 14 bits which is much lower
7:36
than 32 bits. Right? So that's step
7:38
number one. And you will see that in
7:40
step number one there is this
7:43
WGMMA which is also called as warp group
7:45
level matrix multiply accumulate. So
7:48
what is this warp? Uh so that is
7:51
essentially warp group level matrix
7:54
multiply accumulate and it performs
7:57
matrix multiply accumulation operation
7:59
using group of warps. So it's a Nvidia
8:01
GPU terminology where a warp is a
8:03
collection of threads right. Uh so no
8:06
need to worry about this. This is diving
8:08
very deep into the Nvidia GPU
8:12
linguistics but warp level matrix
8:14
multiply accumulate or warp group level
8:17
matrix multiply accumulate performs MMA
8:19
operations efficiently within Nvidia
8:22
GPUs that's all you need to know right
8:24
now essentially even if you uh
8:26
understand this segment by saying that
8:29
the matrix multiplication inputs are
8:32
stored in low precision accumulation
8:34
that much is enough and then in step
8:37
number to what we do is we promote this
8:39
low precision or this low precision
8:41
accumulated values to a higher precision
8:43
we promote it to CUDA course. So after a
8:46
certain interval typically after 128
8:49
elements that is denoted by NC over here
8:52
uh the partial low precision
8:54
accumulations are promoted by promoted
8:56
we mean they are copied to high
8:58
precision registers in the cuda course.
9:00
So after 128
9:03
elements the low precision accumulated
9:06
elements are transferred to a high
9:08
precision koda core right and that's
9:11
shown by this dark purple dot over here.
9:13
This is actually called increased
9:15
accumulation
9:16
precision and that's why if you see the
9:19
this schematic is titled increased
9:21
accumulation precision. It just means
9:23
that we are promoting from a low
9:25
precision tensor core. Uh we are
9:27
promoting from a low precision tensor
9:29
core to a high precision coda core. So
9:32
the first step is low precision MMA
9:34
accumulation in tensor core. And the
9:36
second step is promotion to higher
9:38
precision coda code. Now this arrow over
9:41
here which I have highlighted right now
9:42
it shows the promotion to Kyoda course
9:45
and it makes sure that partial sums are
9:47
stored in high precision
9:49
memory and after the promotion the
9:51
partial results are accumulated in full
9:54
floating point 32 precision on COD
9:56
course
9:57
right then these these blue dots over
10:01
here are the scaling factors because we
10:03
get these scaling factors during
10:05
quantization right and then we need to
10:06
dequantize the quantized element
10:09
elements back into their original shape.
10:11
So these are the scaling factors used
10:13
during deontization. Remember we looked
10:15
at quantization in the previous lecture
10:17
where uh quantization just takes me from
10:22
uh let's say floating point 32 to
10:24
integer 8. So this is quantization and
10:27
descaling is going or upscaling is going
10:29
into the reverse direction where we
10:31
dequantize back to the original numbers.
10:34
So this dequantization is just
10:36
represented by this
10:39
uh uh dark blue dot which you can see
10:42
over
10:43
here. That's it. This is the high
10:46
precision accumulation or increasing
10:49
accumulation precision which is a
10:51




***



technique implemented by deepseek. So
10:53
now that you know what goes on behind
10:55
increasing accumulation precision, it's
10:57
quite simple to understand the schematic
10:59
right. Tensor cores accumulate
11:01
quantities in low precision maybe
11:03
floating point 14 which is not good that
11:05
leads to numerical errors. So we
11:07
periodically transfer the low precision
11:09
accumulation to a high precision coda
11:12
core that is shown by this dark purple
11:14
dot over
11:15
here. That's all there is to understand
11:18
about increasing accumulation precision.
11:20
Okay. After this understanding the next
11:23
thing which we'll try to understand is
11:25
mantisa or exponents. So let's go to
11:27
understand this section right now.
11:30
So until now we have covered three major
11:33
quantization techniques which deepseek
11:35
has implemented and that is mixed
11:37
precision framework fine grain
11:39
quantization and increasing accumulation
11:42
precision. We'll now look at two minor
11:45
techniques which they implemented. First
11:46
is mantisa or exponents and second is
11:49
called as online quantization. So let's
11:51
get started with looking at mantisa over
11:54
exponents.
11:56
Uh first of all if you remember quant if
12:00
you we have taken a look at this blog on
12:02
quantization by Martin Gutendors and
12:05
here you will see what mantisa means and
12:07
what exponent means. So if you look at
12:10
mantisa and exponent they take up
12:13
different number of bits and if this
12:15
number
12:17
340 625 is to be represented exponent.
12:21
So this mantisa covers how many decimals
12:24
and exponent covers the first aspect. So
12:26
it's 2 to 1 over here multiplied by 1 +
12:30
570 3125 etc.
12:34
So every floating point representation
12:37
has a sign exponent and a mantisa as
12:39
shown in the given figure. And the
12:41
critical thing to understand is that uh
12:45
mantisa actually covers the precision
12:47
and
12:48
exponent uh tells us about the dynamic
12:51
range. So there are two types of format.
12:54
So E4 M3 and E5 M2 where we have four
12:59
exponent bits and three mantis suba bits
13:02
and E5 M2 where we have five exponent
13:05
bits and we have two mantis suba bits.
13:07
So what does having more number of
13:09
exponent means? It means that our range
13:11
is larger and what does having more
13:13
number of mantisa point means? It means
13:15
that our precision is larger. So
13:18
remember that mantisa controls precision
13:20
and exponents exponent controls the
13:23
dynamic
13:24
range. In this blog in the first lecture
13:27
we have also seen what dynamic range
13:29
actually means. It just the range which
13:30
is represented. So this is controlled by
13:33
the exponent value and mantisa just
13:36
controls the precision which is distance
13:37
between the two neighboring
13:39
values. U so if you look at the E5 M2 it
13:44
has five exponent bits and it has two
13:46
multis sub bits right. So it has larger
13:48
numerical range but lower precision and
13:50
E4 M3 it has smaller numerical range but
13:53
it has a higher precision. So usually
13:56
prior to deepsek what people did is that
13:58
they did the forward pass using E4 M3
14:01
which has a higher precision and they
14:03
did a backward pass using E5 M2
14:07
uh which has a lower precision. So the
14:09
forward pass had a higher precision but
14:11
lower dynamical range and the backward
14:13
pass had lower precision but a higher
14:15
dynamical range. Deep seek changed that.
14:18
Deep seek said that instead of using E4
14:20
M3 for forward pass and E5 M2 for my
14:23
backward pass, I'm going to always use
14:25
E4
14:26
M3. So D4 Deep Seikk chose E4 M3
14:30
uniformly for all their passes for the
14:32
forward pass as well as the backward
14:34
pass. and they argued that it works in
14:36
their favor because of the fine grain
14:38
quantization. So let's say if we have
14:40
this set of numbers right without fine
14:43
grain quantization we'll have the
14:46
numbers divided by the largest number
14:48
which is an outlier and these numbers
14:51
become very small uh and they lose
14:54
precision significantly uh since only
14:56
three mantis subabits are available in
14:58
E4
15:00
M3 but in this case with fine grain
15:03
quantization the numbers don't become as
15:07
small as these so Even if we have three
15:09
mantisa bits is usually fine for us the
15:12
numbers are still very accurately
15:14
represented because now we divide the
15:16
number into two groups right and each
15:18
group has a separate scaling
15:20
factor. So even if we scale uh with the
15:23
scaling factors no number becomes very
15:25
small and numbers can still be
15:27
accurately represented in E4 M3 without
15:29
losing their precision. So if we use E4
15:32
M3 it it will work since we are doing
15:35
fine grain quantization.
15:38
So the main idea here is that without
15:40
fine grain quantization some numbers
15:42
become very small and if we are using E4
15:45
M3 throughout these numbers will lose
15:46
precision but with fine grain
15:49
quantization no number becomes extremely
15:51
small right so even if we have three
15:52
mantis sub bits it's usually fine for us
15:56
uh
15:57
everywhere so what deepsek mentioned is
15:59
that uh each element effectively shares
16:04
the group's exponent bits and this is
16:06
because Because of the way fine grain
16:08
quantization is implemented allowing
16:10
higher accuracy within each
16:12
group. This significantly expands the
16:15
effective precision without the need for
16:17
extra exponent bits. So we don't need
16:19
five exponent bits in this case because
16:22
uh we are doing fine grid
16:25
quantization. So the reason I'm
16:27
deliberately not getting into too much
16:29
technical depth here is because this is
16:32
not as important as the other techniques
16:35
which we have covered. The only thing to
16:37
note here is that instead of using E4,
16:39
M3 and E5 M2, DeepSc always used E4 M3.
16:43
So they did not use higher number of
16:45
exponents. More exponents leads to more
16:47
dynamical range. But deep did not need
16:50
that because they are using fine grain
16:52
quantization which solves that
16:55
problem. Okay. So that's the main idea
16:57
behind Mantisa over exporates. And the
17:00
last technique which we are going to see
17:01
is something called as online
17:03
quantization.
17:05
So before understanding online
17:06
quantization, we first need to
17:08
understand what is delayed quantization.
17:10
So as I mentioned to you before,
17:13
quantization needs the scale factor,
17:15
right? Quantization needs the scale
17:18
factor. We divide every value with the
17:20
largest absolute value and then we
17:22
multiply with let's say 127 if you want
17:24
to convert from floating point 32 to 2
17:26
in 8. Now
17:29
uh what is meant by delayed quantization
17:32
is that the scale factor used for
17:34
quantizing the current tensor is derived
17:36
from past iterations which means it's
17:38
derived from historical information not
17:41
the present value. This means that we
17:43
are using the past maximum value to
17:45
scale the current current tensor and
17:48
that can lead to inaccuracy. So if the
17:51
current tensor has significantly
17:53
different range it can lead to overflow
17:55
or underflow. So the main idea is that
17:57
in delayed quantization the scale
17:59
factors are taken from past iterations.
18:01
But online quantization what it does is
18:04
that deep seek solution to this is
18:06
online quantization. So they calculate
18:08
the scale factor in real time based on
18:10
the current tensors data. So the maximum
18:13
absolute value is computed on the fly.
18:16
Um so this does not lead to the issue of
18:18
overflow or underflow. So here I have
18:21
given an example of online quantization
18:24
uh versus delayed quantization. So in
18:26
delayed quantization if you use a
18:28
previous scaling factor you can have
18:30
ranges which exceed. So if you're using
18:32
floating point with a range of 240 this
18:35
value here 480 uses a past scaling
18:38
factor. So that leads to overflow that
18:40
leads to an error and delayed
18:42
quantization. But in online quantization
18:44
we use the present scaling factor. So
18:46
here nothing is overflowing or
18:48
underflowing. we are still in that FP8
18:50
dynamical range. So the main idea here
18:53
is that the scaling factors are
18:54
calculated on the fly in real time. They
18:57
are not based on past historical
18:59
information as is traditionally done in
19:01
delayed
19:03
quantization. So these are the two
19:06
remaining minor techniques which we saw.
19:08
So overall in this lecture we covered
19:10
three techniques. The first technique
19:12
which we covered is increasing
19:14
accumulation precision which is
19:17
mentioned over here. The second
19:19
technique which we covered is mantisa
19:21
over exponents. And the third technique
19:22
which we covered is online quantization.
19:25
All of these techniques supplement the
19:27
first two techniques. So deepse
19:29
implemented all five of those mixed
19:31
mixed precision framework fine grain
19:33
quantization increasing accumulation
19:35
precision mantisa or exponents and
19:37
online quantization. And now I hope that
19:40
with these three lectures on
19:41
quantization you will be able to
19:43
understand this section. So section 3.3
19:46
on floating point a training. When I
19:48
first read about this section, it is
19:49
very difficult to understand what
19:51
exactly is going on here because they
19:53
have so many schematics with different
19:55
colors with different shapes, so many
19:57
different terminologies that it causes
20:00
confusion. That's why I broke this down
20:03
into three lectures. So in these three
20:05
lectures, we have covered section 3.3,
20:08
section
20:09
3.3.1, section 3.3.2 completely. So
20:14
until this part, we have covered online
20:15
quantization. That's the last thing we
20:17
saw. So if you scroll up and look at the
20:19
deepseek table of contents, we have
20:22
covered FPA training in these series of
20:24
lectures which is a very important
20:26
aspect of deepseek technical
20:28
implementations. Of course, the
20:29
architectural part is equally important
20:31
which we have covered in the previous
20:33
lectures. But even the floating point A
20:35
training has its own special place in
20:38
the way Deepseek implemented
20:39
quantization, the reduced memory storage
20:41
and the increased computational speed.
20:44
All of these techniques which DeepSync
20:45
implemented especially
20:48
um the multi head latent attention the
20:51
mixture of experts the multi-token
20:52
prediction and now quantization all of
20:55
this actually came together which gave
20:57
them huge performance gains in
20:59
pre-training and also reduced their
21:01
inference costs it just amazing how
21:04
deepseek accumulated all of these
21:06
techniques together and they have
21:07
publicly released this information from
21:09
which we can learn so much the main
21:12
thing which I like about deepse seeks
21:13
architecture is that they did not
21:15
probably invent something entirely new.
21:18
They already built upon the work which
21:20
other people have done and then they
21:22
made fine-tuning to it. They made
21:24
efficient gains to those techniques and
21:27
that we saw with multi- latent attention
21:29
mixture of experts multi-token
21:31
prediction also with quantization. The
21:33
common thing is they already took some
21:35
knowledge which people have worked on
21:36
and they added extra efficiency gains
21:40
and extra improvements to that already
21:41
existing knowledge.
21:43
Thanks everyone. With this lecture we
21:45
have completed the part on quantization
21:47
and I look forward to seeing you in the
21:49
next lectures.
Shorts
