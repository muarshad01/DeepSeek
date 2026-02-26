#### Integer and Binary Positional Encodings (PE)

1. Integer Positional Encoding (IPE)
2. Binary Positional Encoding (BRE)
3. Sinusoidal Positional Encoding (SinPE)
3. Rotary Positional Encoding (RoPE)

***

* 5:00

#### Why do we need positional embedding or positional encoding?
* The main reason is that the position at which a word comes in a sentence is very important to the context of the sentence.
* __Input embedding__ = Token embedding (Capture Semantic Information) + Positional embedding

```
The dog chased another dog.
```

* After coming out of the attention block, we want the model to understand the context of the sentence. We want the model to understand that the first dog who is chasing is different from the second dog who is being chased. Right?

* [Huggingface example](https://huggingface.co/blog/designing-positional-encoding)


#### Integer Positional Encoding (IPE)

```
The dog chased the ball. It couldn't catch it.
```

* 20:00

#### Binary Positional Encoding (BPE)

* Lower indices oscillate fast between positions. Higher indices oscillate slow between posoitons.

***

* 25:00

***

* 30: 00
  
moment? You can also pause this video here for a while. One of the main problems of
30:51
integer positional encoding is this graph. You see the issue with this graph
30:56
is that there are discontinuities, right? The values of integer position they're inputting are discrete. They are
31:03
either zero or one and they lead to discontinuities in the resulting vectors. So during pre-training it
31:09
becomes difficult for the LLM optimization. Remember what's going to happen is that we don't know these positional values from the start. We are
31:17
going to pre-train these values along with all the other parameters of the LLM. So during pre-training it becomes
31:23
difficult for the language model to deal with these jumps. It becomes difficult for the langu language model to deal
31:30
with these jumps. So back propagation becomes harder. So ideally what would be the
31:36
best for us is that we have a graph similar to what we are seeing over here but the values are smoother. Can we have
31:43
a graph like this? Uh can we have a graph like this where the values are smoother and the values are not
31:49
discontinuous? And immediately that suggests an intuition for us, right? If we want
31:55
values which are smoother but not discontinuous like this, what if we have sine functions or what if we have cosine
32:02
functions instead of these discontinuous values and that eventually brings us to
32:07
the next type of encoding which is sinosidal positional encoding which we are going to cover in the next lecture.
32:14
So so far what we have seen is that we have seen that there are two types of positional encodings which we have seen
32:21
in today's lecture. First is integer positional encoding. The major issue with integer positional encoding was
32:26
that the values were not constrained. So the positional encoding values were far higher with magnitude than the token
32:33
encoding values and they diluted the information given by the token encoding which is not good. Token encodings
32:39
capture the semantic meaning and I want that information to be retained. So ideally we want these values to be
32:45
bounded. So then we started thinking that if we want these values to be bounded why don't we have a binary uh
32:52
encoding for every every value. So if it's 200 let's represent it with a binary
32:57
representation. And once we represent it like this it constrains the values but it opens the door for new
33:04
interpretations which means that now what I can do is that I can have two variables. I can have my position on the
33:11
x-axis uh which is the position of that token in the context and then on the y-axis I
33:19
can have index number one index 2 3 4 5 6 7 8 right these are the different
33:24
indexes and then I can actually plot how different indexes change with my
33:29
position. So the lower indexes index 1 and two change very fast whereas higher indexes actually change very slow. This
33:36
indicates that the lower indexes are actually capturing something different than the higher indexes. Right? The
33:42
lower indexes capture quick changes in my positional encoding values. Whereas
33:47
the higher indexes don't capture these quick changes. We are going to see what every index means and why are we looking
33:55
at this visualization later when we look at syanosal and rotary position encodings. But just remember that this
34:01
graph opens a door for new visualization index and position and we can plot this
34:06
oscillation frequency. The main issue is that this these are not continuous values right these are completely
34:13
discrete values which means that there are jumps there are discontinuities which are not
34:19
good for back propagation that's where sinosidal encodings come into the picture where
34:24
what cyanosidal encodings actually propose is that why don't we have why
34:30
don't we make this smooth and continuous like this so then this will also be differentiable and then back propagation
34:36
will be just much more easier Here these sinos cyanosidal encodings were exactly what was proposed in the attention is
34:43
all you need paper. What you see over here are these sinosidal positional encodings. So from this paper it's very
34:49
difficult to understand why this is exactly proposed. But to truly understand this you need to go through
34:55
this whole exercise of first there was integer positional encodings then we have uh then we have binary position
35:02
encodings then we saw that this is fully discontinuous. So then why don't we make it continuous. Then why don't we come up
35:08
with cinosidal positional encodings. So once you go through positional encodings in this manner things become a lot more
35:14
easier than just going through this formula which is very tough and very difficult to understand if you don't
35:20
have any background. So these people they just have two paragraphs two or three paragraphs on this. But to unpack
35:25
this we are going to do this in two to three lectures. Rotary positional encodings came after this actually. So
35:32
in the next lecture we are going to look at sinosidal positional encodings that will be a full lecture because it's a
35:37
very detailed concept and after that we are going to look at rotary positional encodings which will eventually help us
35:44
understand how multi latent attention was mixed with rotary positional encoding. We'll also need to see why
35:51
people now use rotary positional encodings much more commonly than sinosodon positional encoding. Thanks a
35:57
lot everyone for attending this lecture. uh positional encodings is one such topic which is very non-intuitive.
36:04
So it's very difficult to at least for me it was quite difficult to understand this formula only when I went through
36:10
this step by step understanding of let's start thinking from first principles and how my positional encodings have been
36:17
derived if I were to derive this first I may have started with integer then binary then I would have moved to
36:23
sinosidal and then when we learn sinosoidal and when we truly understand it that provides us a very nice clue or
36:31
a hint to go to rotary positional encoding. which we are going to see in tomorrow's lecture. Thanks everyone and
36:37
I look forward to seeing you in the next lecture.










