#### problem with the self attention mechanism is and
* The artist painted the portrait of a woman with a brush.
1. painting a woman with a "brush"
2. painting of a "woman with a brush"

***

* 10:00

* Self attention can only capture a single perspective in a given input sequence. It cann't capture multiple perspectives.

***

* 20:00

| Self Attention | Perspective |
|---|---|
| 1 Self Attention | 1 Perspective | 1 Context Vector Matrix | 
| Multiple Self Attention | Multiple Perspective | Multiple Context Vector Matrices |

***

* 25:00

#### Implementing a 2-head attention (step-by-step)

1. Input Embedding
* Example: (11 X 8), where d_in=8
2. Start with a single W_q, W_k, W_v
* Example: (8 X 4), where d_out=t
* Output is (11x4) for query vectors, key vectors, value vectors
3. Split W_q, W_k, W_v into multiple heads
* Example: (8 X 2) (W_q1,W_q2), (W_k1,W_k2), (W_v1,W_v2)

$$\text{head-dim} = \frac{d_{out}}{num ~of ~heads} = \frac{4}{2} = 2$$

***

* 35:00

6. Computing attentin weights for each head
* Scaling - softmax - causal attantion - Dropout
7. Merge the contex matrix for two heads

***

* 45:00

multi-ad attention mechanism we no llm just has a single head we have multiple heads so that each head can capture
46:03
different perspective so this is the whole stepbystep procedure of how we go from the self attention mechanism to the
46:11
multi-head attention mechanism which was the main purpose of today's lecture now what I want to show you is that I want
46:17
to show you a very quick demonstration of uh how of visualization of these
46:24
attention heads so what we are going to do is that we we are going to take a pre-trained large language
46:30
model we are going to pre we are going to take a pre-trained llm so this is going to be a Burt model and it will
46:37
have a bod directional attention so causality is not implemented so every token will look at previous tokens and
46:43
also the tokens after that so this is pre-trained which means it has already been optimized on a huge amount of data
46:50
what we'll do is that we'll pass our input sentence we'll pass our input sentence to this pre-trend llm and
46:56
what's the input sentence the artist painted the portrait of a woman with a brush we'll pass our input sentence to
47:02
this pre-rain llm and then what we'll do is that we'll peek into the different
47:08
uh attention heads and we'll see what every attention head essentially gives
47:13
us so remember that when we see the code there will be two parameters there will be layer there will be layer and the
47:20
head so what layer essentially means is that an llm architecture has multiple
47:26
Transformer blocks and each Transformer block has multiple attention heads so when we look at different layers it
47:33
means different the different Transformer block and when we look at head it means that which head we are in
47:39
in a particular layer so for the purposes of demonstration we are only going to look at layer number three
47:46
which is essentially the third Transformer block and in this layer number three we are going to look at
47:51
attention head number three and attention head number eight what does it mean attention head number three and
47:57
attention head number eight the pre-trend llm which we are looking at we have 11 attention heads so the output
48:04
Dimension is split into 11 different parts and then every every attention head will essentially get D out divided
48:11
by 11 and then we are going to look at the attention weights Matrix such as
48:17
this uh such as this for each head and that is going to tell us that when we
48:24
look at wom for example what is prioritized by different attention heads
48:29
so let's quickly jump into the demonstration right now so the uh package which I've download downloaded
48:36
over here is bwiz and then what I'm simply doing is that I've loaded the pre-trained uh
48:43
pre-trend model over here and I'm just showing a visualization for this sentence the artist painted the portrait
48:49
of a woman with a brush and first I want to show you for layer number three and
48:54
we'll see for layer number three and essentially head number three so if you go into layer number three and head
49:00
number three and if you hover on to woman let's see so if you hover on to woman you will see that the maximum
49:06
attention is given to brush if you see on the right hand side the maximum attention if you trace this line You'll
49:13
see that the maximum attention is given to brush and we can also confirm this so
49:19
um in this in this code I've essentially plotted the different uh attention
49:25
scores which are given for or woman and I've have taken a screenshot over here
49:30
so if you look at layer three and head number three and if you take the query as woman here you can plot the tokens
49:39
for which the maximum attention weight is given so the maximum attention weight or score is given to brush for layer
49:45
three and head three but now let's go to layer three and head number eight so if I go to layer three and head number
49:51
eight right now you'll see that when you see wom the maximum attention is now given to Port
49:57
and that is again confirmed over here if you see layer number three and head number eight maximum attention is given
50:03
to Portrait so this might indicate that here the attention of the woman is given
50:09
to brush right so that might mean that here we have the second
50:14
visualization um so let me take a
50:24
oops um after this loads let me take a screenshot of uh that second
50:30
visualization yeah so it it seems that the attention between woman and brush is
50:35
the maximum right so it seems that head number three which we saw over here thinks of this interpretation or this
50:42
perspective because it seems in the second perspective the woman holds a brush in her hand and the attention
50:48
between woman and brush is the maximum in this head whereas if you look at head number eight the attention between woman
50:55
and brush is very low so it seems that this head has recognized that maybe the woman is not holding the brush but the
51:01
woman is just present in the portrait so it might mean that this second head thinks that this
51:08
perspective uh is more strong so it it maybe decodes this
51:16
perspective so here you can see this is the direct proof that a pre-trained Transformer or pre-trained llm rather
51:23
has different attention heads and each attention head can essentially uncover a different meaning the the head number
51:30
three which we saw over here uncovers this meaning that maybe the woman holds a brush in her hand whereas head number
51:37
eight uncovers this meaning that maybe it's just a portrait of a woman and the woman might not be holding a brush but
51:43
the artist might just be painting the portrait of a woman so in this Hands-On demonstration we just saw that different
51:50
attention heads can capture different perspectives and that's the whole aim or the whole purpose of the multi-head
51:56
attention mechanism I hope I've been able to explain why what's the intuitive need
52:03
that we need to go from the self attention mechanism to the multi-head attention mechanism this lecture was
52:08
specifically dedicated to introducing the intuition behind multihead attention mechanism in the next lecture what we
52:15
are going to do is we are going to do actual calculation using mathematical numbers so we are going to start with a
52:21
given sentence we are going to assume some numbers and we are going to uh apply multihead attention in practice
52:28
but I did not want to directly jump to this lecture without giving you an intuition for why we move from self
52:33
attention mechanism to multi-ad attention mechanism this is the core building block of why llms work and it's
52:41
also the core building block which deep seek figured out that we need to modify this block itself uh to make it a bit
52:48
better so although multi-ad attention has a lot of advantages it does have some disadvantages in some in terms of
52:54
storage space and comput efficiency which are mitigated or reduced by key
52:59
value cash and further reduced by multihead latent attention so that's why we have this important milestone in our
53:05
way we cannot understand KV cach or multi-head latent attention without understanding multi-ad attention itself
53:13
so after the next lecture we'll directly move to Key value cach and then multi-ad latent attention which is the First
53:20
Fundamental innovation in deep seek so stay tuned till that time and I hope you are making notes along side these
53:27
lectures are a bit dense and I'm deliberately making them a bit longer so
53:32
that everything is explained to you this won't be a lecture series of two to three lectures I'm planning it to make
53:38
it 35 to 40 videos of lecture Series so that ultimately you really understand
53:44
the nuts and bols of how deep seek is constructed but to go to that stage it's
53:49
important for us to be on the same page with the building blocks thanks a lot everyone and I look forward to seeing
53:55
you in the next lecture

















