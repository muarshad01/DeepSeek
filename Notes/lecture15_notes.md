#### Sinusoidal Positional Encoding (SinPE)

$$
\begin{aligned}
    PE_{(pos, 2i)}   &= sin\bigg(\frac{pos}{100000^{\frac{2i}{d_{model}}}}\bigg)\\
    PE_{(pos, 2i+1)} &= cos\bigg(\frac{pos}{100000^{\frac{2i}{d_{model}}}}\bigg)\\
\end{aligned}
$$

* (pos, index)
  * Event index = $$2i$$
  * Odd index = $$2i+1$$
  * The values live in the range [-1, 1]

* [Attention Is All You Need - 2017](https://arxiv.org/abs/1706.03762)

***

* 15:00

#### EXAMPLE GPT-2
* Context size = 1024
    * The range of "pos" variable is [1, 1024]
* Embedding dimentions = $$d_{model}$$ = 768
    * The range of "i" or index variable is [1, 768]

***

* 20:00

* SinPE avoid the discontinuous nature that was there with BPE.
* SinPE values are smooth, continuous, and also differentiable.
* Thats helps LLM training routine.

***

* 25:00

* __Property 2__: Linear relation between two encoded positions.

***

* 30:00

  
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



***

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













