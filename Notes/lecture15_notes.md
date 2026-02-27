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
* Thats helps to a much stable LLM optimization routine.

***

* 25:00

* __Property 2__: Linear relation between two encoded positions.

***

* 30:00

***

* 35:00

* Relative positional encodings are just rotations of each other.

* rotations ensure that relative ship shifts map to fixed angular differences, which then translate into predictable, learnable attention patterns - like focusing more on nearby words.

#### What's the main problem with sinusoidal embedding?
* One major issue is that we add these encodings directly to token embeddings. This can __pollute the semantic information__ carried by token embeddings.


* can we instead augment my query and the key vectors itself with positional embeddings?

***

* 40:00


*
*
*
* So think about this for a
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


















