
0:05
Hello everyone and welcome to this
0:08
lecture in the build deepseek from
0:10
scratch
0:11
series. Today we are going to start an
0:15
important topic and that is with respect
0:17
to quantization. So until now with
0:20
respect to building the entire deepseek
0:22
architecture we have covered multi head
0:25
latent attention we have covered mixture
0:27
of experts and in the previous
0:30
um set of lectures we also covered uh
0:35
um the third major architectural
0:39
uh pillar of deep seed and that's
0:41
multi-token prediction. So after
0:44
covering these three major pillars, we
0:46
now move towards quantization and that's
0:49
the final pillar with respect to the
0:52
deepseek architecture. So when I say
0:54
quantization, if you take a look at the
0:56
deepseek version 3 technical report and
0:59
if you scroll down below, you'll see
1:01
that uh we have covered until now the
1:03
latent attention mixture of experts and
1:05
multi-token prediction. So the
1:07
architecture part of this paper is
1:09
finished. Now if you come to the
1:11
infrastructures portion they have this
1:13
FP8 training and uh these major aspects
1:17
of the FBA training are what we are
1:19
going to cover in these sequence of
1:21
lectures. So if you scroll down below
1:24
you'll see that
1:26
uh they have a section which is called
1:29
as FP8 or floating.8 training section
1:33
number 3.3 and that comes up on page
1:36
number 14.
1:38
This section is about the quantization
1:40
which deepseek has implemented and like
1:42
several other sections of this paper if
1:44
you directly start reading this it will
1:47
be a bit difficult to really understand
1:49
what's going
1:51
on. So what I'm going to do is that in
1:53
this series of two to three lectures I'm
1:56
going to explain the five arch the five
2:00
innovations which deepseek implemented
2:02
in their quantization routine. First is
2:04
called as mixed precision framework.
2:07
Second is called as fine grained
2:09
quantization. Third is increasing
2:11
accumulation precision. Fourth is
2:14
mantisa over exponents and fifth is
2:16
online
2:18
quantization. Okay. So when deepsee
2:21
released their paper they had figures on
2:24
each of these sections. So for example,
2:26
if you look at figure number six, that
2:29
is the figure which illustrates the
2:30
mixed precision framework which we are
2:33
going to
2:34
uh which we are going to see in detail.
2:37
This is the mixed precision framework.
2:39
Then if you look at these figures which
2:41
is figure number seven, they have part A
2:44
and part B. So part A is with respect to
2:47
fine grain quantization and part B is
2:50
with respect to increasing accumulation
2:52
precision. If you directly look at these
2:54
figures, it may seem that there are so
2:56
many building blocks over here. And what
2:58
do these different colors actually
3:00
represent? I plan to break it down for
3:02
you so that you'll understand it. The
3:04
third or the fourth part is mantisa over
3:07
exponents and the fifth is online
3:10
quantization. So if you see they have a
3:12
part called mantisa over exponents and
3:15
they have a part called or a section
3:16
which is called as online quantization.
3:20
So the way I'm going to break down these
3:22
lectures is that before understanding
3:24
any of these we'll first need to have a
3:26
brief understanding of what exactly is
3:28
meant by
3:29
quantization. So in this lecture I'm
3:32
going to very simply explain to you what
3:34
quantization means and starting from
3:38
next lectures I'll take you through all
3:40
of these frameworks. So the next lecture
3:43
may be based on these three mixed
3:45
precision framework fine grain
3:46
quantization increasing accumulation and
3:49
then I'll have another lecture on
3:50
mantisa or exponents online quantization
3:53
and then we'll wrap up the quantization
3:55
part. So let's get started with today's
3:58
lecture. U first let me show you the
4:01
main reference which I'm going to use
4:02
today. So quantization I'm going to use
4:05
a blog
4:07
um which is written by Martin Gautam
4:10
Dors. This is probably the best
4:12
reference which I' I've come across to
4:14
understand quantization and uh I'll be
4:17
referring to this blog in today's
4:19
lecture where I explain to you
4:21
quantization and what it exactly means.
4:23
So let's get started with that. Okay. So
4:27
deepse is a large language model right?
4:30
So why does quantization come up in
4:33
large language models and what exactly
4:35
is
4:36
quantization? So if you look at the
4:39
basic building blocks of a large
4:40
language models, we have inputs, we have
4:43
weights, inputs are multiplied by
4:46
weights and pass through activations,
4:48
right? So these activations which we are
4:51
referring to, they are calculated after
4:54
let's say you get the product of the
4:56
inputs and the weights and these
4:59
activations then serve as an input to
5:01






***


the future blocks etc. Essentially what
5:04
I'm trying to say is that there are
5:05
number of parameters in a large language
5:07
model which multiply among each other
5:10
and then we have several summations
5:11
which need to be done. So for example if
5:14
you consider this entire large language
5:16
model if you look at the multi head
5:18
attention block the multi head attention
5:21
block has several matrix multiplications
5:23
right like queries will be multiplied
5:24
with keys transpose. So if you focus on
5:27
each of these matrices, if you focus on
5:29
the queries matrix, if you focus on the
5:31
keys matrix, elements of these matrices
5:34
will be multiplied with each other.
5:36
Similarly, if you have feed forward
5:38
neural network, there are multiple
5:39
parameters which will be multiplied with
5:41
each
5:42
other. The reason I'm telling you all of
5:44
this is because parameters when you look
5:46
at every individual parameter uh it
5:49
occupies a certain memory. So there are
5:51
parameters in token embeddings,
5:53
positional embeddings, multi head
5:55
attention, feed forward neural network
5:57
and if you consider the scale and shift,
5:59
there are parameters in layer
6:01
normalization also. We have the
6:03
parameters in the output layer. All of
6:05
these parameters take up memory. So just
6:08
like uh if you are building a house, the
6:11
house will take up some area, right?
6:13
Similarly, whenever you define a
6:15
parameter, that parameter takes up
6:17
memory. The amount of memory a parameter
6:20
takes up is determined by how we
6:22
represent that parameter. Right? So
6:26
there are several representation
6:27
techniques. The most common by default
6:30
parameters are represented as floating
6:32
32 values or
6:34
FP32. For example, if a parameter is
6:38
3.14159, let's say the way it's
6:40
represented in memory is by these 32
6:42
bit. So if you count these bits one bit
6:45
then we have 1 2 3 4 5 6 7 8 there are
6:50
eight bits over here. So and then
6:52
finally we have 1 2 3 4 5 6 7 8 9 10 11
6:57
12 13 14 15 16 17 18 19 20 21 22 23 so
7:04
23. So if you add 1 + 8 + 23 that gives
7:07
you 32 bits. So in this case which I'm
7:10
showing you the every parameter is
7:12
represented by 32 bits in
7:15
memory. What each of these bits mean is
7:18
slightly different. The first bit
7:20
controls the uh sign. So as is mentioned
7:24
over here in this blog if you
7:26
see yeah the first bit over here
7:29
controls the sign. The second bit or
7:32
these set of bits is called as the
7:34
exponent. And this last set of bits is
7:37
called as the significant or the
7:39
mantisa. So essentially there is a
7:41
simple formula through which how the
7:43
sign the exponent and the mantisa all
7:46
add up together to represent the entire
7:49
number. The number of bits which you
7:52
have in the mantisa in the last part
7:54
controls the amount of precision with
7:56
which you can represent every number by.
7:58
Okay. So at the top here I'm showing you
8:02
a floating point a floating 32-bit
8:05
representation and if you take a closer
8:07
look at the
8:10
um at this blog over here let's see so
8:13
if you take a look at the floating point
8:15
32 representation and if you scroll up
8:18
below you'll see how these sign exponent
8:20
and mantisa work. So sign is actually
8:23
pretty simple whatever the value here is
8:25
you take minus1 raised to that. Uh so if
8:28
it's zero it means it's positive because
8:30
- 1 to0 will be 1. If it is equal to 1
8:33
that's - 1 to 1 so it will be a negative
8:35
number. So the first bit determines the
8:38
sign right. The second bit which is the
8:40
deter which is the exponent. So it's in
8:43
this case it will be 2 to 1 and the last
8:46
part which is the mantisa which will be
8:48
2 to0 plus
8:50
5701
8:52
5703125. So that will give us
8:55
3.14625. That's float 16 bit. For float
8:59
32-bit, it will be much higher
9:01
precision. So if the same parameter
9:04
right now is represented as a float 16
9:06
bit instead of a float 32bit, it will
9:09
just take 16 bits in memory. So it will
9:12
take less amount of memory. But the
9:14
downside of that is that we'll have a
9:16
lesser number of bits in the mantisa
9:18
part. Right? So that means we will have
9:20
a lower precision. So that same number
9:22
will be represented with a very high
9:24
precision over here. And that same
9:26
number will now be represented with a
9:28
very low precision in a float 16 bit.
9:31
Right? So essentially what I'm trying to
9:34
say is that every parameter takes up
9:37
memory. And the amount of memory a
9:38
parameter takes up deterine is
9:40
determined by how it is represented. If
9:43
it is represented by 32 bits, it takes
9:45
more memory and the parameter can be
9:47
represented in a higher precision. If
9:50
it's represented by 16 bit, it takes up
9:52
less memory and that parameter is
9:54
represented in lower
9:57
precision. Now here I have shown that if
10:00
we are representing a parameter as 64
10:02
bits and we have a large language model
10:04
with 70 billion parameters, the amount
10:07
of memory the all the parameters take is
10:10
70 billion multiplied by 64 divided by
10:13
8. The reason we are dividing by 8 is
10:15
because 8 bits equal to one bite.
10:19
So if you represent every parameter of a
10:21
70 billion model as a 64-bit
10:24
floatingoint number, it takes 560 GB of
10:27
space. If you represent every parameter
10:29
as a 32-bit floatingoint number, it
10:32
takes two 280 GB of space. And if you
10:35
represent every parameter as a 16 bit
10:37
number, then it only takes 140 GB of
10:40
space. So the if you reduce the number
10:44
of bits a parameter is represented by
10:47
the amount of memory that parameter can
10:49
take will significantly reduce and
10:52
that's exactly what is meant by
10:54
quantization. So if you look at the
10:56
definition of quantization quantization
10:58
aims to reduce the precision of a
11:01
model's parameter from higher bit widths
11:03
to lower bit widths. So essentially if a
11:06
parameter is represented as 32bit
11:08
quantizing means we represent that same
11:11
parameter in lower bits right now. So if
11:13
a parameter is 64 bits we'll represent
11:15
it with 32 bits or we can represent it
11:18
with 16 bits. That's what's meant by
11:21
quantization. So you might be thinking
11:23
why would we do this? So I understand
11:25
that we save up memory but what about
11:27
the precision which is lost. So it turns
11:30
out that there are in several operations
11:32
which are involved in the LLM
11:34
architecture. It's fine if we lose out
11:36
on some precision as long as overall the
11:38
precision or the accuracy is maintained.
11:41
So it turns out that we don't need to
11:43
represent everything as floating point
11:45
32. Even if we represent a parameters
11:48
with floating point 16 for example, it
11:50
might still lead to good results. So
11:52
then why would we not do that? That's
11:54
why people convert floating point 32
11:56
into lower bit representations while
11:59
still we maintain the accuracy of the
12:02
large language model. That is what is
12:04
meant by
12:05
quantization. So one image which I love
12:07
in this blog is that if you take a look
12:09
at the original image
12:12
uh whereas if you take a look at this
12:14
quantized image if you look at it from
12:15
far both of them look kind of similar
12:17
right but if you zoom in you'll see that
12:19
the original image is much clearer
12:21
whereas the quantized image is quite
12:23
pixelated. So that's shown in this inset
12:26
right? If you zoom in here you'll see
12:28
that this inset is much more sharper
12:30
than the quantized. The reason is
12:33
because in the quantized image it's only
12:34
made up of eight
12:36
colors, eight distinct colors. Whereas
12:39
the original image is made up of much
12:41
larger set of colors. So this this
12:44
illustration just shows that if you do
12:46
quantization, you take up less amount of
12:48
memory, you take up less amount of space
12:50
because now this quantized image has
12:52
lesser number of uh uh colors or pixels
12:56
which have been used to represent the
12:57
image. But we kind of represent the
13:00
original image itself. It's not like we
13:02
are getting a huge amount of reduction
13:03
in the performance. That's why we do
13:06
quantization because we also get memory
13:08
reduction while performance is not
13:10
hampered that much. It's important to
13:12
keep in mind that performance slightly
13:13
degrades because the parameters lose
13:15
their precision but overall it does not
13:18
hamper the performance that much.
13:21
So now I just want to uh mention or
13:25
quickly explain some quantization
13:27
techniques which are very commonly used
13:30
in
13:31
u which will also come up in the next
13:34
lecture on how deepseek implemented
13:36
quantization. So if we want to go from
13:38
floating point 32 to floating point 16
13:41
this is how it is done right. So
13:43
floating point 32 we can represent a
13:46
much larger range of values over here.
13:48
Whereas if you go to floating point 16
13:50
we can represent a much smaller range of
13:53
values. So for example in floating point
13:55
32 the minimum value is - 3.4 into 10 to
13:58
38. The maximum is 3.4 into 10 to 38.
14:02
Whereas in floating point 16 the minimum
14:05
value is - 65504. The maximum is
14:09
65504. And you will also see that the
14:11
precision is reduced. So
14:14
3.14159 here there are so many decimal
14:17
points right but if you consider
14:19
floating point 16 the precision is
14:21
reduced and the range is also
14:23
reduced. Uh but this schematic shows how
14:27
you can quantize a number from floating
14:29
point 32 to floating point 16. In the
14:32
next lecture you'll also encounter
14:34
something called BF16. That's
14:35
essentially brain float 16. So BF16 has
14:39
the same number of bits as floating
14:40
point 16 but it maintains the range. So
14:43
you see in floating point 32 to floating
14:46
point 16 we reduce the range right but
14:49
from floating point 32 to BF16 the range
14:52
is maintained the BF16 also the minimum
14:56
is - 3.4 into 10 to 38 maximum is also
14:59
3.4 4 into 10 to 38 but the number of
15:02
bits are still equal to 16 in BF16. So
15:04
that stays the same between BF16 and
15:07
let's say
15:09
FP16. And the third is integer 8. So
15:12
that's the lowest form of representation
15:14
where the minimum and maximum go from
15:16
minus 127 to 127 and only eight bits are
15:20
represented over here. That's why it's
15:21
called as integer 8. So here every value
15:24
is essentially an integer. We don't have
15:26
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

