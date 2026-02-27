
***

* 5:00

***

* 100

#### Sinusoidal Positional Encoding (SinPE)

$$
\begin{aligned}
    PE_{(pos, 2i)}   &= sin\bigg(\frac{pos}{100000^{\frac{2i}{d_{model}}}}\bigg)\\
    PE_{(pos, 2i+1)} &= cos\bigg(\frac{pos}{100000^{\frac{2i}{d_{model}}}}\bigg)\\
\end{aligned}
$$

* (pos, index)
  * Event index = 2i
  * Odd index = 2i+1


the values are sign of position divided by 10,000 to 2 I divided by the model
10:46
dimension. If it's an odd index then the value for the positional encoding for that index is cosine of position divided
10:53
by 10,000 to 2i by the model dimension. So let's say that we are considering for
10:59
example um we are considering GPT2 we are considering GPT2 and that has the model
11:07
dimension of 768 and I want to find the positional encoding for position
11:14
number let's say I want to find it for position number two and now every position remember
11:20
every position will have uh uh a vector every position so
11:26
position number two will have a vector whose size is equal to 768. So this is the positional embedding
11:32
vector of position 2. It will have a size of 768. So the I value the index
11:38
will go from 1. So this is let's say I is equal to 1 right up till index
11:45
768. What this formula gives is that it finds what value will be there at every
11:50
index. Right? So at I = if I put I = 1 then it will be position which means
11:57
second position and so let's start from I equal to0 actually. So my value at zero position
12:05
will be sine of position which is 2 divided by
12:12
10,000 raised to 2 into I. So my I is zero right? So it will be zero.
12:19
So my value at position number zero. So let's say I starts from zero and goes to 76
12:26
uh7. So for for if I put I equal to0 then the
12:32
value here will be given by this sign. And if I equal to0 then my positional
12:38
embedding at second position and index equal to 1 will be equal to cos position
12:44
divided by so cos 2 / 10,000 raised to 2 /
12:52
768 because actually my i is equal to zero. So this will still be zero. So then I
12:59
get this value as well um i = 1. Now to find the next values
13:06
what I do is then I put my i equal to 1. So I get my values for
13:11
the second index and I get my values for the third index. So then I'll fill second index and third index
13:17
appropriately. Similarly I'll go right till the very end using this formula. So
13:22
the thing to note is that if you take a look at this 768 dimensional vector the
13:28
even positions the even positions which is I =0 I = 2 I =
13:34
4 I = 0 I = 2 I = 4 these are given by the
13:42
sinosidal or sign formula and if you take a look at the odd positions if you
13:48
look at take a look at the odd positions which is I = 1 then i = 3 then i = 5
13:57
these are given by the cosine formula we'll see why there is a separate formula for the sign for the even
14:03
positions and why there is a separate formula for the odd positions but I hope you understood what are the different
14:09
variables over here there are two variables over here right there is the position variable and there is the index
14:16
variable so the way to read this formula is that you fix a position and you will have a 768 dimensional vector. Now you
14:23
need to decide what's each of those 768 values right those are my 768
14:28
indexes and to get the value at every index I first decide whether that index is even or whether it's odd. If it's
14:35
even I use this formula if it's odd I'm going to use this formula and that's how I fill all my values. So you might be
14:43
thinking where did this formula actually come from right and where did it originate? This formula was proposed in
14:49
the attention is all you need paper. This was the main paper in which the transformer architecture was proposed
14:56
with the attention mechanism. We did not have a recurrent neural network over here. Until this paper was proposed,
15:03
attention mechanisms were always coupled with recurrent neural networks. But this was the first one with which freed the
15:09
attention mechanism from the RNN. Now if you scroll down below, there is a very small section on sinosidal positional
15:16
encoding. You see only three paragraphs where they introduce these two these two
15:21
functions. How to find the positional encoding values at different indices using the s and the cosine functions. So
15:29
this is a these are three paragraphs over here right but to understand it we need to devote this full lecture towards
15:35
it. So first let's actually start exploring these let's start exploring this formula and let's see whether it
15:42
makes sense or not. First we see that it's s and cosine right? So that's why the values lie on a continuous spectrum
15:48
between minus1 and 1 that that much is obvious. But where does this why do we
15:53
have 10,000 to 2i by d model? Let's see. So let us start by visualizing this
16:00
formula. The best way to understand whether this formula works or not is to start visualizing it. So we are going to
16:07
use GPT2 and we are going to use the D model equal to 768. Now GPT2 let's say the
16:15
model dimension is 768. What does this model dimension decide? This decides the
16:21
range of my indexes. So if the model dimension is 768, my index will go from I=0 till I =
16:28
767. And what else do I need? I need to check how many positions or what values this POS
16:35
can take. Right? And remember what will that depend on? Pause this video for a moment here if
16:41
you want to think about that the position will depend on the context size. So GPT2 had a context size of 1024
16:50
which means that the position of any token which I choose can lie between 1
16:55
or you can say 0 to 1023 or if it starts from 1 it can lie between 1 to
17:01
1024. There will be no token whose position is 1025 because it will be out of my
17:07
context size. Right? So I my I goes from 0 to 767 which is the index and my
17:14
position goes from 1 1 to 1 0 to4 right. These are the two variables which we
17:20
always deal with and those are the same variables we dealt with when we looked at the uh binary position encoding.
17:27
Remember in the binary position encoding we had the position on the x-axis and we have
17:33
the index on the y-axis. So let me assume the D model equal to
17:38
768. So the range and the context size is 1024. So range of position variable
17:44
is 1 to 1024. And if my I actually starts from 1, this can be 1 to 768. It
17:49
depends on your convention. So the I the range of I variable I'm assuming to go from 1 to 768. Right? Now what we can
17:57
easily do is that we can write a very simple code to actually plot um uh to
18:03
fix to fix a particular I. So we can fix a particular I over here. Uh let's say I
18:10
fix a particular index and then plot the value of the positional encoding for
18:15
different positions. Right? So for i= 1. So this is my i = 1 now. And you'll see
18:23
on the x-axis I have my different positions. And what am I exactly plotting over here? I'm I'm looking at
18:29
this thousand different positions and I'm plotting their low lowest significant bit. I'm plotting the lowest
18:37
significant bit of my thousand positions. What does that mean? Remember that now every position let's say I have
18:44
position number one. I have position number one that will be a 768 dimensional vector.
18:52
I have my position number two that will be a 768 dimensional vector. Correct? Similarly, if you go right till the very
18:59
end, I have position number 1024. So, this will be
19:05
1024 and it will have 768 different values or 768 different indexes. All of
19:12
these will have 768 different indexes. What we are plotting here is the lowest. So, if the lowest index starts from
19:19
here, this is I equal to 1. We are plotting the last index for all of these
19:24
for all of these positions. And what have we learned before? We learned before that um what was the biggest
19:31
learning for for us? Our biggest learning was lower indexes change more frequently and higher indices change
19:38
less frequently across positions. Let's see whether that happens here also. What we are doing here is that I'm just
19:44
collecting the index equal to one values for all positions and I'm plotting it. So here you see how quickly the lower
19:51
indexes are oscillating. It's almost it's extremely quick, right? And the reason it's so quick is because this
19:57
will oscillate after every position. After every one position, it will immediately oscillate. That's why you
20:02
see so many oscillations over here. Now what I've done is that I also go a bit deeper and I say if I equal to
20:09
50, I have also collected all the values for I equal to 50 and I plotted them together. So here you see the frequency
20:16
of oscillations now slightly reduces as my index value increases from 1 to 50.
20:23
And the last plotting I have done is for I equal to 150. So I go even one step ahead and I plot I equal to 150. I
20:30
collect this value for all the different positions and I plot them together. So here you see now the frequency reduces
20:37
by huge amount. So if you go from i = 1 to i = 50 to i = 150. What is it that
20:46
you notice? The main thing which you notice is that lower indices oscillate fast
20:52
between positions. Higher indexes oscillate slow between positions. That is exactly what we have seen for binary
21:00
positional encoding. We had seen that lower indexes change more frequently, higher indexes change less frequently.
21:07
This is the same thing which we observe here. Lower indexes oscillate fast between positions and higher indexes
21:12
oscillate slow between positions. Now this is one proof that this formula actually works. What this
21:19
formula does is that it captures the intuition of binary positional encoding. How do we know that? Because we got the
21:26
same result like right lower indexes oscillate faster higher indexes oscillate slower. that intuition remains
21:33
the same whether we use binary or whether we use this formula. So that's awesome. The second thing which this
21:39
gives me is that sinosidal encodings are now continuous. They lie between minus1 and one. So you see these these curves
21:46
now they are not discrete like we had observed for binary positional encoding.
21:52
All of these curves were completely discrete and jumpy. Right? For sinosidal positional encodings I have these curves
21:58
which are now smooth and continuous and also differentiable. Okay. So the main advantage of sinosidal
22:05
encoding or cinos cyanosidal embedding over binary encoding is that the positional graphs are continuous as
22:12
opposed to being jumpy and discontinuous and that leads to a much more stable LLM
22:18
optimization routine. Okay. So it solves the problem which was there with integer
22:23
positional encodings. So if you look at this formula and if this formula seems too difficult to understand, forget
22:29
about this and go back to binary positional encoding. In binary positional encoding, it's very easy to
22:34
visualize this formula and how the values change across different positions and for different
22:41
indexes. What this s and cosine formula at its core, it's actually doing the
22:46
same thing. We are saying that the lower indexes oscillate faster, the higher
22:51
indices oscillate slower. That's why this denominator has 10,000 to 2 I. So
22:58
the index has to be there in the denominator because the lower indexes need to oscillate faster. The higher
23:04
indexes need to oscillate slower. So one more way to look at this formula
23:10
is that the frequency of oscillation the frequency of oscillation is given by so
23:16
if you look at this formula right let me rub this
23:22
uh and let's take a closer look at this formula. This formula can also be written as first let's look at the even
23:28
positions right it can be written as sin of omega * p where p is my position
23:34
right so what is omega omega is 1 / 10,000 ra to 2 i by d
23:42
model this is my frequency of oscillation this is my frequency of
23:50
oscillation and here you can clearly see that the i comes in the denominator.
23:55
Right? So for lower indices the frequency of oscillation will be higher and for higher indexes the frequency of
24:02
oscillation will be lower. That's an intuition for why this I has to come in
24:07
the denominator. And why do we have 10,000 here? That's an experimental choice. The reason is because the model
24:13
dimensions will be close to 1,000. So it just makes sense that if we have a 10,000 over here um my frequencies will
24:22
go on decreasing eventually for I equal to 150 200 300 etc. If I change this
24:27
value of 10,000 maybe my frequencies will die down very fast or very slow which I don't want. This 10,000 just
24:34
gives an optimal range so that for I equal to 1 the frequencies are very high and for I equal to 150 200 300 500
24:41
they'll slowly die down and then ultimately become flatter. That's an understanding of this formula
24:48
intuitively. Right? So until now what we have seen is that the sign and sinosidal positional encodings have the same
24:55
intuition as binary positional encodings. Lower indexes oscillate faster. Higher indexes oscillate slower.
25:01
Cool. The second thing which we saw is that sinosidal positional encodings avoid the discontinuous nature which was
25:08
there with binary positional encodings. They are smooth, continuous and differentiable.
25:14
What that helps us is that the smooth continuous and differentiable nature which is there which you see in these
25:20
plots that helps my LLM training routine. That was one of the major issues with binary positional encoding.
25:27
So you might still be thinking right all of this now seems fine but one thing is still unclear for me and that thing is u
25:36
why do we really have a s and a cosine over here? What's the reason for this? that was not explained until now. So one
25:45
thing to notice is that even indices have a sign formula or odd indices have a cosine formula. Why do we need to do
25:51
this? What's the reason for this? Um and actually if we understand this reason,
25:57
it truly sets the stage to understand rotary positional encodings.
26:03
So remember that this reason will ultimately set the stage for R O which
26:09
is rotary positional encodings which is the main thing which we want to understand to eventually see how rope or
26:15
R O rotary positional encoding is mixed with latent attention. So let's start understanding this reason.
26:22
The key to understanding the reason for the use of s and cosine is to understand
26:30
one key property which we need when we construct positional encodings. When we construct positional encodings, we want
26:38
to have a linear relation between two encoded positions. This means that
26:43
whatever formula we decide to use for positional encodings whether it's s cos integer binary encodings the
26:50
relationship between positions should be mathematically simple which means that if we know the encoding for position p
26:57
it should be straightforward to compute the encoding for position p + k why
27:03
because imagine you are a transformer and you want to learn positional patterns if there is already an
27:11
underlying pattern pattern which we have encoded it will be much easier for you as a transformer to learn the positional
27:17
patterns among different tokens. So that's why when you look at this attention paper you have you can
27:23
see here that the authors have mentioned we must inject some information about the relative or absolute position of
27:30
tokens in the sequence. So when we construct position encodings, we should construct them smartly so that if you
27:37
have the positional encoding vector at a given position, you can easily find out the position encoding vector at other
27:44
positions also mathematically so that the transformer can also learn this mathematical
27:49
property. Keep this in mind. We have a desire to have a linear relation between
27:54
two encoded positions. With this in mind, we'll see that having the sign and
28:00
cosine actually help us to achieve this. Given the positional encoding of one token at position pos, the sign and
28:07
cosine make it much easier to find the positional encoding of another token at
28:12
another position, let's say pos or position plus k. Which means that if we have the positional encoding at a given
28:18
position, having the sign and cosine makes it much easier to find the encoding at another position. Let's see
28:25
how and why this is possible. Right? So for the purposes of this visual demonstration, I'm assuming that I'm at
28:31
a given position P and I'm looking at this formula. So let me let me pull down
28:38
this formula from here. Uh this is the positional encoding formula for
28:44
different indexes. Right? So here in this formula, what I'm going to do is that I'm going to choose I equal to 1.
28:50
So if I = 1, I'm going to look at indexes 2 I which is equal to 2 and 2 I
28:56
+ 1 which is equal to 3 at a given position. So if you look at the given
29:03
position at index 2, what's its value? Its value will become sine of my given
29:09
position divided by 10,000 to 2 / D.
29:15
Correct? And this value will become cos of my given position divided by 10,000
29:22
to 2 / d. So what I have drawn over here is that let's say
29:28
um let's say this is this value here position divided by 10,000 to 2x d is
29:33
equal to theta. So if this is equal to theta then my sin theta will be the y1
29:40
which is shown over here and my cos theta will be the x1. So this value is
29:45
now represented the sin theta is represented by y1. So y1 is the positional encoding value at position
29:51
two. Y1 is the positional encoding value at position 2 and x1 is cos of theta.
29:57
Right? So that will be the positional encoding value at position number three. That's what I've shown in this plot. Y1
30:04
corresponds to sin of theta. So it's at positional encoding at position 2. And
30:11
x1 corresponds to cos theta. So it's positional encoding for a position or i equal to 3 for index
30:19
equal to two for index equal to 3 and here we have index equal to two for y1
30:25
that's what's mentioned here y1 represents the sinosidal positional encoding for a given position at index 2
30:31
at index 2 and x1 represents the sinosidal positional encoding for a given position p at index number three.
30:39
Okay. Now let me give you a puzzle. What if I want to find the
30:46
um what if I want to find the position encodings for a position P
30:52
+ K. So currently I'm at a certain position P right which also I'm calling POS. So let's say I'm at a position P +
31:01
K. Okay. How can I find the values the sinosidal positional
31:06
encoding values for I= 2 and for I = 3. So I know this Y1 and X1 for my given
31:13
position. But what if for position P + K? How can I find I = 2 and I= 3. I'm
31:19
only looking at these indexes. So let's say we want to find
31:24
the cinosidal positional encoding values for a position shifted by K. So it turns
31:30
out that you can simply find these values by just rotating my initial vector. So this was my initial vector
31:36
v_sub1. Correct? Let's say I rotate that by a factor of
31:41
theta_1. Let's say I rotate that by a factor of theta_1 where theta_1 is now
31:46
omega * k. Right? So if you do theta + theta 1, my theta + theta1 will now
31:53
become omega * p + k. Right? Um so this
31:59
is theta + theta 1. So what will be sin of um theta + theta 1? It will be sin of p
32:08
+ k divided by 10,000 to 2 by d and cos
32:15
of p + k / 10,000 to 2 + 1 / d. And if
32:21
my i is again equal to 1, this will just be two and this will be
32:27
three. So what this means now my y2 corresponds to this value, right? My y2
32:34
is now sin of my y2 is now sin of theta +
32:40
theta1. My y2 is now sin of theta + theta1. And my x2 is cos of theta +
32:47
theta1. So my x2 is cos of theta + theta 1. and my y2 is sin of theta + theta
32:55
1. So to find the values for index = 1
33:01
or for index= 2 and 3, all we have to do is rotate the given vector v1 by an
33:06
angle of theta 1 equal to omega k where k is the amount of positions which I need to shift
33:12
by. And then y2 represents the positional embedding vector of position p + k at index number two. and x2
33:20
represents the positional encoding vector of position p + k at index number three. So you see earlier the in the
33:28
first vector y1 represented the uh positional embedding vector at index 2
33:34
x1 represented at index 3. But for the position P. Now if you want to shift the position by K, all you have to do is you
33:41
have to rotate my vector by an angle equal to omega K where omega is given by
33:47
this and I will be equal to 1. And so in my given vector I just take the X component of my new rotated vector that
33:54
will be the positional encoding vector of the new position at index 3. and my y component of the new rotated vector will
34:01
be the positional encoding vector of position p + k at index equal to
34:07
2. So what this means is that v_sub_1 and v_sub_2 are just rotations of each other. Which means that if we know the
34:14
positional encoding value for a given position, if we know the positional encoding value for a given position, to
34:21
find the positional encoding value for a new position, all we have to do is rotate my vector.
34:27
This means that relative position encodings are just rotations of each other. Now do you see why we have sine
34:33
and cos? We have sine and cos because the rotations will become possible. Without s and cos rotations will not
34:39
come into the picture. The cosine is there because that corresponds to the x coordinate. The sign is there because
34:46
that corresponds to the y-coordinate. And because positional encodings here
34:51
are rotations of each other, it satisfies this property that there is now a relation between encoded positions. Which means that if we have
34:58
the encoding vector for one position to find the position encoding for another position, we just rotate the initial
35:05
vector. That's beautiful, right? Instead of just adding or subtracting, we rotate
35:11
to get the new position encoding vector. For a new position, we rotate from an initial position. That's where the
35:17
rotation comes into the picture and this is the idea which we are going to expand
35:22
later when we learn about rotary positional encodings. So the reason we encode
35:28
alternate indexes by s and cosine is that to rotate a vector we need both s
35:34
and cosine terms otherwise rotation will not be possible. The s and cosine in the
35:40
formula ensure that relative positional encodings are rotations of each other.
35:45
Okay. And that makes sure that there is a relation between different positions which the transformer can later
35:52
learn. That's why when this formula was released, it actually solves a number of things, right? First, it makes sure that
35:59
instead of having this jumpy jumpy values for binary positional encodings,
36:04
now the s and cosine values and this 10,000 which is an experimental choice
36:09
make my frequency values or not frequency values make my oscillation
36:14
smoother, right? So, it helps the LLM stabilization routine. Secondly, my s
36:22
and cosine which are there now ensure that um relative position encodings are
36:27
just rotations of each other. That's a very important thing to note. Without this formula, it would
36:34
have been impossible to extract this fact that relative positional encodings are rotations of each other. So when
36:41
this formula was released, when this formula was released u a number of a huge number of
36:48
researchers and organizations actually used this to build foundational language
36:53
models and that's why it became very popular. Uh one more thing to note is
36:59
that rotations ensure that relative ship shifts map to fixed angular differences
37:04
which then translate into predictable learnable attention patterns. That just means that we are injecting some
37:10
information from our side that the positions are actually related to each other and the transformer later learns
37:16
this injected information. So we are talking about cyanosidal positional encodings as if
37:23
they solve all of our problems, right? Then why do we even need rotary positional encoding? What's the main
37:29
problem with cyanosidal embeddings? Let's see that. Now one of the main issues with sinosidal positional
37:36
encodings is that positional embeddings are directly added to token
37:41
embeddings. Although the magnitude of the positional embeddings is small, the
37:47
very fact that we are adding positional embeddings to token embeddings pollutes the semantic information which is
37:52
carried by token embeddings. Ideally, we want the semantic information to be carried by token embeddings to be
37:58
preserved when we go into the transformer block. But in sinosidal positional encodings we add the
38:05
positional encoding value to the semantics to the semantic information and that actually pollutes the semantic
38:13
information. So one major issue is that these encodings are directly added to
38:19
token embeddings which pollutes the semantic information carried by token embeddings.
38:25
If you carefully look at the attention mechanism, you'll see that the attention
38:30
mechanism is that place where the influence of one token on another is
38:35
quantified by the queries and the key matrix. Right? You multiply the queries multiplied by the keys transpose and
38:42
that gives you the attention scores. This is that place where the position of a token really matters because for one
38:49
query I look at all the other keys and I find the attention scores. So instead of
38:54
adding the positional embeddings to my token embeddings, can we instead augment
39:00
my query and the key vectors itself with positional embeddings? So think about this for a
39:07
moment, right? Why do we want to encode positional encodings? Why do we want to do that? Because we want to let my
39:14
transformer architecture know that certain tokens are at different position than other tokens. Okay. Uh what do you
39:22
mean you want to know? What do what do you mean you want to let your transformer know? It means that when I
39:27
compute the context vectors from my token embedding vectors, tokens which are at different
39:34
positions should have different context vectors. Okay? To have different context vectors, you need different query and
39:40
the key vectors for different positions. Right? Then why don't you make sure that the query and the key values itself the
39:48
query and the key vectors itself these vectors change for
39:55
different positions which means that can you augment the query and the key vectors
40:02
such that their values are different for different
40:08
positions. So that's one one thing to note. Okay, that's one realization to
40:14
note that uh we don't necessarily need to add things to our token embeddings. We don't
40:20
necessarily need to make the changes here. We can make changes in the queries and keys. Second thing to note is that
40:27
let's say if I have one vector, right? If I have one vector and I want to
40:32
include the information of the position to this vector. Currently, we are just adding another vector to this, right? So
40:39
that might change the magnitude of my current vector. Instead of adding another vector to this vector, what if
40:45
we just rotate this vector? What if we just rotate this vector? So if you rotate the vector,
40:52
individual values will change, but the magnitude of the vector itself will remain the same. That's the second
40:57
learning. The first the first learning is that instead of making changes in my token embedding by adding positional
41:04
embeddings, what if so let's say I have my different tokens here, right? I have five tokens. Token 1 2 3 4 5. Let's say
41:12
I augment this, augment this, augment this, augment this, augment this based on their positions. And now when I say
41:18
augment, I'm not going to add another vector to this. I'm going to take each of these and I'm going to rotate them.
41:24
And the amount of rotation will depend on the position of the token. What this
41:30
will do is that it will do two things. I'll not contaminate my token embedding vector because now I'm encoding
41:36
positional level information in the queries and keys. So my semantic information is retained before I enter
41:41
the transformer. Second thing is that I'm not changing the magnitude of my these
41:47
individual query vectors and these individual key vectors. I'm just rotating these vectors
41:53
to capture the information of the position. And the amount of rotation will depend on which position the token
42:00
comes in. So the magnitude of these queries and the keys vectors remain the same. So we will preserve the original
42:06
vector magnitude by merely rotating the vector. So why don't we add positional
42:13
encodings to query and key vectors which encoded relative token which encodes relative token information. Uh and
42:21
instead of adding instead of adding it directly we are going to rotate this. Now these two ideas the first idea that
42:29
why don't we do the position operation at the query and key level and second instead of adding a vector why don't we
42:35
rotate these vectors these two ideas led to rotary positional
42:40
encodings. Um so the main drawback of sinosidal positional encoding was that token embeddings are polluted which we
42:47
don't want. How does rotary embedding solve that? In rotary embeddings we are going to look at augmenting the queries
42:53
and the key vectors. We are not going to touch my token embedding vectors at all. Second thing which we are going to do is
42:59
that the query and the key vectors their magnitude won't be changed. We are just going to rotate them to
43:06
uh to signify or to influence the position of the token. We are not going
43:11
to change the magnitude of the vector. These two things ultimately led to
43:17
rotary position encodings or rope which is the key idea which deepseek exploited
43:23
along with multi head latent attention. In fact, Llama also uses
43:28
rotary positional encodings and many modern LLMs uses rotary positional encodings to prevent the token embedding
43:37
contamination. That brings us to the end of today's lecture. Today we covered sinosidal positional encoding which
43:43
really serves as a building block to rotary positional encodings. Without understanding sinosoidal positional
43:48
encodings, you would not have understood that what is the reason for not using sinosidal encodings which we covered at
43:55
the end. And secondly, why do we have cos and s? What do cos and s actually do? Here we built up the built up the
44:03
intuition that cos and s are needed because rotations would not have been possible without cos and s. And why do
44:10
we need rotations? We need rotations because one positional encoding should depend on another positional encoding.
44:17
We should inject that information mathematically so that the transformer can learn that
44:22
information. I hope this formula when you see this formula does not feel as scary to you now as it might have done
44:29
earlier. So there are two variables here, the index and the position. The index depends on the model embedding
44:35
dimension and the position depends on the context vector size. The lower indexes oscillate the fastest.
44:41
The higher indices oscillate the slowest. That property is retained which was the same with binary encodings. It's
44:47
just that here the oscillations are completely differentiable which was not the case
44:53
with binary encodings. So thanks a lot everyone. I hope you are enjoying these lectures.
44:59
These are very dense deep lectures because position encodings is one topic which I think many people have not
45:05
understood this formula. I don't know how many people have completely understood the reason for this formula.
45:11
The 10,000 is still an experimental choice. But it's very important to understand two things. Why this I comes
45:18
in the denominator like 10,000 to 2 I? Why not in the numerator? This is because lower indexes should oscillate
45:24
faster than the higher indexes. Why do we have s and cosine? Sine and cosine are there?
45:30
because without them uh we cannot have the rotations between different position
45:35
vectors which are essential because there should be some relation between one position and another position. Now
45:41
if you know the positional encoding for one position you just rotate it to find the position encoding for another
45:47
position. That way this formula becomes a lot more intuitive. And now the stage is set for us to completely understand
45:54
rotary positional encoding. which will be the uh topic of our next lecture. So
45:59
please make notes as we are going along these lectures and after the next lecture on rope we are ready now to
46:06
fully understand how deepsek integrated multi head latent attention with rope.
46:12
So thanks a lot everyone. uh make detailed notes as you follow along and I look forward to seeing you in the next
46:18
lecture.


