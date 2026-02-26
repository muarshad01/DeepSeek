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

```
The dog chased the ball. It couldn't catch it.
```

* 20:00

#### Binary Positional Encoding (BPE)

* Lower indices oscillate fast between positions. Higher indices oscillate slow between posoitons.

***

* 25:00


Fifth index every 16. Sixth index every 32. Seventh index every 64. And the
25:26
eighth index after every 128 positions. So that's why you'll see that since we are only showing 11 positions over here
25:34
index these indexes won't oscillate these indexes won't oscillate
25:40
in the example which I've 64 to 75 which I've shown because they oscillate every
25:46
16 32 64 and 128 positions.
25:51
So if you plot the oscillation frequency versus the index position, you'll see that the index position one has the
25:56
highest frequency and then the oscillation frequency actually goes down. Right? So the lower indexes
26:02
actually capture immediate changes across the different position
26:08
values. Whereas the higher indexes capture small moving changes across the
26:14
different position values. Which means that position number one and position number eight. Now uh so the higher index
26:21
position will not change unless we jump 128 positions. So if you have a sentence and
26:29
nearby tokens are or there are nearby tokens which are the same um which have
26:36
the same word but they are spaced out by a factor of three or four which means they are close by. their differences are
26:43
mostly captured by the um lower indexes because the lower
26:50
indexes capture faster oscillations. So one more graph which is
26:56
extremely important which I want to show you now is this particular graph right um and let's try to understand this
27:03
graph in detail. So you'll see that on the x-axis I have shown the position um
27:08
and that can be any position here 64 65 66 etc. And for every position now we
27:15
have eight indexes right index one index 2 index 3 4 5 6 7 8. So the way to
27:21
interpret this graph is look at a position and every position has eight different indexes. Now 1 2 3 4 5 6 7 8
27:28
which are the values between 0 and 1. What this particular plot shows is that how the different index values
27:34
oscillate. So you see the first index oscillates very very fast right after every after every position change it
27:42
oscillates. Here we saw this directly even with one position change it
27:47
oscillates. So that's why the frequency of oscillation of index one is the highest. The frequency of oscillation of
27:54
index 2 is slightly lower. It oscillates after every two positions. Index three
27:59
oscillates after every four positions. Index four oscillates after every eight
28:05
positions. Index number five oscillates after every 16
28:10
positions. Index number six oscillates after every 32 positions. Index number seven oscillates
28:18
after every 64 positions and index number eight oscillates after every 128
28:23
positions. I hope you have understood this graph because this is the same graph which
28:28
we'll be seeing in the cyanosidal positional embeddings as well. Okay. So what we have shown in this graph is that
28:35
we have considered two factors the position of the token and the index number. These two factors will later
28:41
show up in cyanosidal positional encodings and rotary positional encodings as well.
28:47
So higher the main conclusion from these so far is that the higher indices or I
28:52
should say the lower indices. So this should be lower indexes over
28:58
here change more frequently suggesting fine grained encoding which means that fast changes in nearby positions and the
29:06
lower index and the higher indexes. So here I should say higher indices higher indexes change less frequency meaning
29:13
less frequently meaning course encoding meaning course encoding uh and that means that they don't
29:21
capture minute changes in positional encoding they capture broad level changes I'll explain the intuitive
29:27
meaning of this when we come to sinosidal encoding in the next lecture but for now just remember that lower
29:33
indexes uh lower indexes change more frequently and let Let me actually just change this
29:39
in front of you so that uh we are all able to see
29:45
this.
29:51
Oops. So what I want to do over here is that I want
29:57
to change this to lower because this should be lower indexes change more
30:03
frequently and this should be higher. which would be higher indexes change less frequently and then I'll again add
30:10
a box around this. All right. Now it seems to be correct. So it seems that with integer
30:16
positional embedding we have solved the issue of the uh with binary positional
30:22
encoding we have solved the issue with integer positional encoding. Right? We have values which are now bounded. They
30:27
are either zero or one. And of course because every position has different
30:32
values, we take care of the first problem which we thought about that different positions should have different values that needs to be added
30:39
to the token embedding vector. So what's the main problem with integer encoding? Can you try to think about this for a
30:46
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








