***

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
