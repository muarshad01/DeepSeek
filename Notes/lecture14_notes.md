#### Integer and Binary Positional Encodings (PE)

1. Integer Positional Encodings (IPE)
2. Binary Positional Encodings (BPRE)
3. Sinusoidal Positional Encodings (SinPE)
3. Rotary Positional Encodings (RoPE)

***

* 5:00

#### Why do we need positional embedding or positional encoding?
* The main reason is that the position at which a word comes in a sentence is very important to the context of the sentence.
* __Input embedding__ = Token embedding + Positional embedding

```
The dog chased another dog.
```

* After coming out of the attention block, we want the model to understand the context of the sentence. We want the model to understand that the first dog who is chasing is different from the second dog who is being chased. Right?

* [Huggingface example](https://huggingface.co/blog/designing-positional-encoding)

***

* 10:00

```
The dog chased the ball. It couldn't catch it.
```

***

* 15:00


zero which means that these values are usually small right on the other hand if
15:36
I have a context length of 1024 let's say I'll have these positional embedding
15:42
values which are very large. So if you add a very very small value to a huge
15:47
value, what will happen? The effect of this smaller value will get completely vanished and it will be mostly dominated
15:54
with this positional information. That's not good at all. Token embeddings play a very important role. Token embeddings
16:01
capture the semantic meaning of words and I don't want that information to be diluted. I don't want that information
16:08
to be lost. Um so if the integer positional encoding
16:15
values can be very large right so we add a very high positional values to the
16:21
very small token embedding values. What this does is that this heavily pollutes
16:26
the semantic information which is captured by token embeddings and uh that defeats the whole
16:33
purpose of token embeddings because we capture the semantic information which is then passed to the transformers.
16:38
Right? What does capturing semantic information means? It means that words which are similar such as let's say
16:47
kitten words which are similar will lie together in vector space whereas words
16:52
which are dissimilar like if there is a a plane here that will lie somewhat
16:59
different from these other vectors. So token embeddings encode semantics. So now if you take these token embedding
17:05
vectors and you add a positional embedding vector suddenly which is a huge value which is a huge value these
17:13
token embedding vectors will get their information will get diluted
17:18
um they won't play such an important role which is not good for us we want this information also to be retained so
17:25
ideally what we want along with the fact that I want to tell the transformer that
17:31
this dog is different than this dog so They should have different values, different positional values. I also want
17:37
to make sure that the positional encoding values need to be constrained and they should not be very
17:42
high. So now think about this right. What can you do to make sure that the positional encoding values are
17:48
constrained hopefully between zero and one let's say. So I'll use the same
17:53
logic. Let's say I have these positions 200 2012 2013. How can I represent them in a
18:00
vector so that every value of the vector is constrained hopefully less than one? How
18:07
will I do this? Again, think about this for a moment. I'm going to pause here for a while. So, this is where binary
18:14
position encoding actually comes into the picture. The main idea is that
18:20
integers can be unbounded, right? They can take very large values. So if I want
18:26
these values to be constrained, why don't I represent this represent every value by binary numbers. What that means
18:34
is that let's say instead of 200, so the position value for the first dog is 200,
18:40
right? What if I represent it as a binary representation whose size is equal to 8, which is the exact same size
18:46
as the token embedding or the token encoding vector. So now you see that if
18:52
I'm going to represent 200 with the binary encoding value then suddenly the
18:57
representation becomes 1 1 0 0 1 0 0 0 right and then each value here is
19:04
constrained to be less than one. It's either 0 or 1. So it's the order of magnitude now of
19:11
the token embedding vectors. The order of magnitude of the token embedding vector and the position embedding
19:18
vectors are around similar. That's great for us. This is exactly what we wanted. Right? So what we'll do here is that if
19:26
we have 200, 2012, 203 etc. all of them will be
19:32
converted into binary representations. So let's see what that actually means. If you have let's say
19:40
you are looking at an input sequence whose positions go from number 64 to 65 to 75. The way each of these tokens
19:48
would be represented. So the way their positional embedding vectors would be represented with will be like this. If
19:54
you start if you look at 64 it will be represented like this. If you look at 65
19:59
it will be like this. Let me use a different color. If you if you see 66 it will be like
20:06
this etc. So every position here will be represented with a eight dimensional
20:12
vector. Now I want you to observe some very key things over here which will play a very very important role when we
20:19
are looking at sinosidal position embeddings and eventually rotary
20:24
positional encodings. One key thing to realize here is that there are two things right. The first is the position
20:30
itself. Here the position is 64 65 66 67
20:36
and within each position I have actually index right. So within each position I have index number
20:44
one, index number two, index number three, index number four, index number
20:49
five, index number six, index number seven and index number eight. So whenever I'm going to refer to index or
20:56
indexes in this lecture and in subsequent lectures of positional encodings, it means these digits right.
21:03
So this index is called as the least significant bit LSB and this first index is called as
21:11
the most significant bit MSB. And I have also mentioned that this
21:17
is LSB and this is MSB. And I also mentioned that over here if you look at 64 we have index 1 2 3 4 5 6 7 8. Index
21:27
1 is called the LSB. Least significant bit. Index 8 is called as the MSB. Now I
21:33
want you to notice a couple of observations over here. Take a look at the lowest index over here which is this
21:39
index number index number one. And let's see how it oscillates. Right? We'll
21:44
start from the top. We'll start from 64 and take a look at this lowest index. You'll see it goes from 0 to 1, 1 to 0,
21:51
0 to 1, 1 1 to 0, 0 to 1, 1 1 to 0, 0 to 1, 1 1 to 0, which means that it's
21:56
oscillating the fastest. Right? Now let's take a look at the second index here. So it's the same for first two
22:04
values 0 0 then it changes 1 one. So its oscillation frequency is slightly
22:10
lesser, right? After every two values it's changing. Then let's look at the third
22:17
year which is shown by the yellow. If you take a look at the third one for the
22:23
first three values or for the first four it's the same. So for the first four it's the same then for the next four
22:29
it's the same then for the next four it's the same etc. So what I'm trying to show here is that lower indexes
22:36
oscillate fast between positions. I you should understand what this sentence mean. What does this mean?
22:42
Lower indexes oscillate fast with respect to positions. What this means is that lower indexes means let's say we
22:49
look at index number zero. Index number zero oscillates the fastest, right? It quickly oscillates as I move from
22:57
different positions. Whereas the if you move to higher indexes, they don't oscillate too much at all. In fact, take
23:03
a look at the seventh index. The seventh index is is same across all. Right? So it does not oscillate at
23:10
all. Take a look at the third index. The third index oscillates after every four positions. The third index oscillates
23:17
after every four positions. But the first index oscillates after every position. So it oscillates the fastest.
23:23
Why am I telling you this? Because later we are going to see that um the lower indexes oscillate the fastest. Right? So
23:30
they capture fast changes among different among positions and the higher
23:35
indexes oscillate very slow. So for the higher indices they do not
23:41
capture fast changes uh across the different positions. This is the same
23:46
concept of indexes and positions which ultimately led to sinosoidal position embeddings. Right? So there are few
23:53
observations which I have made here. The first index oscillates between let me show this with a different color. The
24:00
first index oscillates between zero and one with every new position. As we can see over here, the first index
24:07
oscillates between 0 and one. Um, the first index oscillates between 0 and one
24:13
for every new position. That's done. That's I've shown this with the
24:18
red color over here. Okay. The second index oscillates between 0 and one with every two positions which is shown by
24:25
this purple color here. It oscillates after every two positions. This is also done.
24:32
The third index oscillates between zero and one after every four positions. So this third let me change or let me uh
24:40
clean some color over here. So third index oscillates between 0 and one after every four positions.
24:47
Right? So this is also done. Similarly in the eth index oscillates between 0
24:53
and 1 with every 128 positions. That's why it does not oscillate from 64 to 75.
24:58
This eighth index over here. This seventh index oscillates between 0 and one every 64 position. So now let us
25:05
write the oscillation frequency for all of these positions. Right? The first index oscillates
25:12
between 0 and one every one position. Second index every two position. Third index every four. Fourth index every 8.
25:20
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






