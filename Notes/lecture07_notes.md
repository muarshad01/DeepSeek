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

* 50:00

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



















