### MLA, MoE, MTP

### 3.3 FP8 Training

* Mixed precision framework.
* Fine-grained quantization.
* Increasing accumulation precision.
* Mmantisa over Exponents
* Online quantization.

* [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)

***

* 10:00

* Quantization aims to reduce the precision of a model's parameter from higher bit-widths to lower bit-widths.

***

* 15:00



decimal places.
15:29
So if you see the three types or the
15:30
four types rather we have FP32, we have
15:33
FP16, we have BF16 and we have int 8. So
15:37
already if you take a look at the deep
15:38
se the first schematic which they have
15:41
in the quantization section, you'll see
15:43
things like BF16, you'll see things like
15:46
floating 8. Now you should be able to
15:48
understand what FP32. FP32 means
15:51
floating 32 representation. FP8 means
15:54
floating point. Here I showed you
15:55
integer 8, right? FP8 is similar to FP16
15:59
but we go one one uh one further level
16:02
below. So what I showed you here is FP16
16:05
right? FP8 is one more level below where
16:08
we go from minus 127 to 127 and allow
16:10
floating points. Uh then FP we have FP8
16:15
BF16
16:17
um I showed you BF 16 already that's
16:20
grain float
16:21
16 then there is also FP32 which also I
16:25
showed you. So now when you see these
16:27
schematics you should be able to
16:28
understand what FP8 means, what BF16
16:31
means, um what FP32 means etc. Those are
16:35
the quantization
16:36
levels. Now the way quantization is
16:39
actually done is that if you have a
16:42
number uh which is represented uh in
16:45
floating point 32 let's say these are
16:47
the set of numbers which are represented
16:49
in floating point 32 and I want to
16:51
represent these numbers in integer 8. So
16:54
they will be from minus 127 to 127.
16:56
Right? The way it's represented is that
17:00
um in my given set I first take the
17:03
maximum value I divide all of these
17:06
numbers with the maximum value. So if
17:08
you have a number x I divide it with
17:10
alpha and then what I'll do is that I'll
17:12
multiply it with
17:14
127 because that's the maximum value
17:17
which can be taken in integer 8. So what
17:20
that does is so if you have this - 7.59
17:23
this will be -
17:25
7.59 divided by 10.8 into 127 and I then
17:30
I take the closest integer to this. So
17:32
that will give me the representation of
17:36
- 7.59. Similarly all of the numbers
17:39
which are in floating.32 are now brought
17:41
back into u integer 8.
17:46
So this calculation which I showed you
17:48
here we have to divide with the maximum
17:50
number right this will play a very
17:52
important role when we are going to look
17:53
at an innovation in deepseek which is
17:55
called as
17:57
uh fine grain quantization. So to
18:01
understand fine grain quantization
18:02
you'll need to understand what I'm
18:04
explaining right now. If you have
18:06
floating point 32 numbers and if you
18:08
want to convert them to any quantized
18:10
let's say integer 8 you first divide all
18:12
of these numbers with the largest value.
18:14
So you divide all of the numbers with
18:16
the largest value and then you multiply
18:18
with 127 because that's the highest
18:21
value let's say which integer 8 can
18:24
take right that's the way you quantize
18:27
from floating point 32 to integer 8. So
18:30
this division of the highest value is
18:31
very important here. So that's where you
18:34
get that broadle scale and then you
18:36
bring it down. So we have the broad
18:37
level floating point 32 and then we
18:39
bring it down to integer rate 8. These
18:41
many concepts I believe are enough for
18:43
you to understand the next lectures in
18:46
quantization which are to follow. So
18:48
once you have this basic understanding
18:49
of why we need quantization because it
18:51
reduces the memory usage while still
18:54
maintaining the precision and the you
18:57
should also know the different
18:58
representations such as what is meant by
19:00
floating point 32 which means we have 32
19:02
bits. Floating 16 has 16 bits and the
19:05
range is also reduced in floating 16.
19:08
BF16 maintains the
19:10
range well whereas we still have 16 uh
19:13
16 16 bits integer 8 has only eight bits
19:16
and the range is much more reduced -127
19:19
to 127 and to actually quantize what we
19:22
do is that if we have floating point 32
19:25
we divide every number with the maximum
19:28
value let's say and then we multiply it
19:30
with 127. So that takes me from this
19:34
uh um this initial floating point 32 and
19:37
brings me back to minus 127 to
19:42
127. This these many concepts will be
19:44
relevant for you to understand mixed
19:46
precision framework and fine grain
19:48
quantization which we'll see in the next
19:50
lecture and after that we'll also see
19:52
increasing accumulation precision
19:54
mantisa or exponents and online
19:56
quantization. All right. So we are now
19:58
slowly making inroads into more deeper
20:01
and deeper aspects of deepseek. So for
20:04
example, if you take a look at their
20:05
table of contents, we have finished the
20:08
whole architectural implementation and
20:10
now we are slowly going into
20:11
infrastructure. In this series, we are
20:13
not going to look at too many hardware
20:15
related aspects. So I'm going to look at
20:17
quantization only and then we'll go to
20:19
the next chapters. Thanks everyone and I
20:22
look forward to seeing you in the next
20:23
lecture.
Shorts


***








