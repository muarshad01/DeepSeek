* 5:00

***

* 10:00

***

* 15:00

* Note that to include positional information, we don't add any vector to the query vector. 
* We effectively rotate parts of the original query vector and hence maintain its magnitude. Thus avoiding the token embedding polluting issue which we saw with sinosidal positional encoding.

***

the matrix by theta and the same rotation matrix is used in rotary
20:20
positional encoding as well. So this is how research advances further research right the sinosidal positional encodings
20:27
came in this 2017 paper and then the rotary positional encodings were later
20:32
derived in 2023 which is 6 years later and here you can see the main they start
20:37






***

* 20:00

out with the same formula which was introduced in the 2017 paper and then
20:43
they augmented this um addition to token embeddings with this image. So if you
20:49
see this image, it's exactly the same as what we have described. You take the query or the key vectors, you divide it
20:55
into into groups of two within one group. So let's say you have one group, you represent it as a vector x1 and x2.
21:02
You rotate that vector by some angle theta which is encoded by the position and the index that gives you the new um
21:09
that gives it you the new position encoded key position encoded query or the key. So you replace the value of
21:15
that group which you had chosen. Then you go to the next group. You replace it again by injecting the position value.
21:21
Then you go to the next group etc. You do this for the first query vector which is at position number one. Then you do
21:27
this for the second query vector. Then you do it for the third etc. That's how that's the geometrical intuition of
21:34
rotary positional encoding. Okay. Okay. So I hope you have understood this much and just another
21:40
last point to note is that I'm saying query vectors here but the same encoding which we have shown here can be done for
21:46
the key vectors also. Okay. Now let's go to understanding a few more intuitive
21:51
details regarding this formula. All right. So if you take a look at this formula over here, you'll
21:57
see that the way we inject positional values is only through this theta. So we
22:03
need to understand what this intuitively means, right? um omega I so let me actually copy this
22:11
let me copy this and bring this down over here to our discussion actually I have already written it down over here
22:18
um so that's fine I don't need to copy copy it so let's start understanding the intuition okay the so this is the value
22:27
with which we rotate omega i into p so theta is is given by my position theta
22:34
is given by my position divided by 10,000 to 2 I divided by D. So it depends on two variables position and
22:40
the index. This D is fixed because that's my embedding dimension. So here I have plotted the a circle here which
22:48
represents the magnitude of this whole quantity. And here I have shown two variables here. If you go from left to
22:55
right the position increases. So the dog chased another dog. This position number one. This is position number two. This
23:01
position number three. Position number four and position number five. And if you go from top to bottom, the index is
23:08
actually increasing. So this is my lowest index. Index number one, index number two, index number three, and
23:13
index number four. So each token I'm assuming as a four-dimensional vector, right? So there are four
23:19
indexes. What I want you to see first is that if you take a look at the effect of position, the rotation magnitude which
23:26
is omega into P varies directly with the position. Okay? So that is shown here.
23:32
Also if you fix any index let's say I fix index number one and as I increase
23:37
the position you'll see that the magnitude of the circle or the circle radius increases right this means as the
23:43
position increases it leads to larger
23:48
rotations. Um so higher positions lead to larger rotations and that's same for all the indexes. If you go to index
23:55
number three, uh if you go to index number three and take a look at the rotation magnitude for every position,
24:01
you'll see that the rotation magnitude increases because the circle radius increases as the position goes on
24:07
increasing. What this shows is that higher positions lead to larger rotations, right? This means that closer
24:13
queries have similar positional encodings. So queries which are closer together they will naturally have
24:19
similar positional encodings because they have similar P and queries which are farther have different positional
24:25
encodings and that makes intuitive sense. This was also mentioned in the
24:31
uh paper which introduced rotary positional encoding which is row former. So if you scroll down below
24:38
um they mention over here that
24:44
okay one can prove that this setting provides a long-term decay property
24:50
which means that the inner product will decay with the inner product will decay when relative position increases. This
24:57
property coincides with the intuition that a pair of tokens with a long relative distance would should have less
25:03
connection. This is the same thing here, right? A pair of query vectors with uh a
25:09
pair of query vectors with a long relative distance
25:14
um will have very different positional encodings because they have different p values. Whereas a pair of query vectors
25:21
which are closer together in position will have similar positional encodings. And that makes intuitive sense to us. So
25:28
if you scroll down below you'll see that uh the second intuition which we observe
25:33
is that the lower index positional values change fast with position and the higher index positional values change
25:39
bit slow with position. Let's see what this means. Right? So if you take a look at the lower indexes you'll see that the
25:47
lower indexes the circle radius increases very fast as the position increases. Whereas if you take the
25:52
highest index the circle radius does not change too much at all. the circle radius is almost fixed which means that
25:59
it's not increasing very fast. So what this shows is that the lower if we take a look at the lower index the rotation
26:07
magnitude changes fast with position high frequency and if we take a look at the higher index which is this the
26:14
rotation magnitude changes slow with position that means it's a low frequency.
26:20
If you remember that's the same conclusion which we had obtained with sinosoidal positional embeddings and binary positional embeddings as well.
26:27
What we had observed was that lower indexes oscillates fast between positions higher indexes oscillates slow
26:33
between position. And since we are using the same formula in rotary positional encoding, we have the same conclusion
26:39
that if you have lower indexes, if you have lower indexes such as
26:44
these this or this, we oscillate fast with positions and higher indexes
26:50
oscillate very slow with positions. So here we have fast oscillations whereas here we have slow oscillations. What
26:56
does this intuitively mean? Um this intuitively means that so lower index
27:02
actually lower index oscillates quickly right this ensures that the model captures small shifts. For example, if
27:09
we have two sentences I just told her the truth versus I told just her the truth. These are different sentences,
27:16
right? I just told her the truth is with respect to time. Recently I told her the truth and I told just her the truth
27:23
which means among all the people I have only her I have told the truth. So the lower index fast oscillations captures
27:29
the change brought about by varying the position of the word told. So here told comes at position number three and here
27:35
it comes at two. So ideally these are very close by positions right. So we
27:41
want that index which can change fast across different positions to capture this change and that's why the lower
27:47
index oscillations becomes very useful for capturing the small shifts uh in
27:52
positions. So I just told her the truth versus I told just her the truth. The word told
28:00
uh in different positions changes meaning and lower index frequencies ensure the model captures the small
28:06
shifts. So whenever you are learning about positional encodings always try to relate it to some intuition like
28:13
this. Uh again when we looked at the first example always ask yourself that why should higher positions lead to
28:19
larger rotations? Because words which are closer together in positions they are more likely to be more related to
28:25
each other. Right? So that's why their positional encodings value should be similar to each other. Whereas words
28:31
which are completely farther apart they should not be that similar. Uh that's why the first point
28:37
makes sense. In the second point the lower index oscillates quickly. The higher index oscillates slowly. So since
28:44
the lower index oscillates quickly, it can capture if there are some small shifts in position which completely
28:50
change the meaning of the sentence. The lower index quick oscillations can capture that. This is also one more
28:56
reason why sinosidal uh this formula works very well because the lower indexes capture the the small shifts and
29:02
the higher indices capture something different which we'll see now. So we understood what the lower indexes
29:08
capture. But if you take a look at the higher higher indexes right which are over here they do not change very fast
29:14
with respect to position. So higher index components ensure that even with large position differences the
29:20
relationship is preserved. What does this mean? So let's say if you have the sentence Einstein developed the theory
29:26
of relativity. This breakthrough reshaped physics. So this breakthrough
29:32
refers to the theory of relativity which which has come several words earlier. So
29:37
higher index oscillations capture these long range dependencies which means that let's say there is a big sentence right
29:44
and there is one uh token at position number one and one token at position number 20 and these are related to each
29:51
other. The higher index the higher indexes are good because the higher indexes don't change too much across
29:57
these two positions and that's why they might capture the fact that these two
30:02
positions are actually related to each other. We cannot rely on the lower indexes for this because the lower
30:07
indexes are quickly changing. Right? What if position number one is here, position number 20 is here or
30:14
here. Let's say position number 20 is here. They change so much. But the higher indexes they change very their
30:22
their frequency is very less. So even for position one and for position 20 they might have similar values. So
30:29
higher indexes have low oscillations which retain or capture the relationship
30:35
between tokens which are very farther apart. So higher index low oscillations capture long range context dependencies.
30:42
That's very important. Okay. So higher index components ensure that even with
30:48
large position differences the relationship is preserved. The relationship between tokens is preserved. Whereas if we only rely on
30:54
lower indexes, the lower indexes oscillate so much that they have no clue that position one and position 20 are
31:01
actually related to each other. So we can rely on the higher indexes for that.
31:06
So lower indexes capture something completely different. They capture small shifts in meaning in the same sentence
31:13
and higher index oscillations capture long range context dependencies. So rotary positioner encoding actually has
31:19
all of this intuition baked within it. We capture fast oscillations. We capture
31:26
slow oscillations. Uh we also make sure that higher positions lead to larger
31:33
rotations and closer positions lead to small rotations which make intuitive sense. So intuitively rotary positional
31:40
encoding makes a lot of sense and more than that it does not change the magnitude of the original query and the
31:46
key vectors. And another important point which is the main difference I feel between rotary and sinosidal encodings
31:52
is that in cinosodal encodings we actually add the positional encodings in the data processor block whereas in
31:58
rotary positional encoding we are taking a look at the multi head attention block and within that we take a look at the
32:04
queries and keys. So we don't dilute the semantic information of token embeddings when we take a look at rotary positional
32:10
encodings. I hope with this you are clear of the visual understanding of rotary
32:16
positional encodings and how they work. This much understanding is enough for us to completely understand how rotary
32:22
positional encoding is actually mixed with multi head latent attention. So now that you have reached
32:29
this stage where you have understood rotary positional encoding in the next set of lectures we are going to see how
32:35
deepsek integrated multi head latent attention with rotary positional encoding. We have seen multi head latent
32:41
attention before and now we saw rotary positional encoding. We'll soon see that
32:46
the traditional multi head latent attention does not directly mix very well with rotary positional encoding. So
32:52
we need to make some changes in the traditional multi head latent attention. Um we'll see why these changes are made.
33:00
we will see the intuition behind these changes and then finally see will the most advanced version of the multi
33:06
latent attention combined with rotary positional encoding. So thanks a lot for sticking with me for the past three
33:12
lectures which were completely on positional encodings. Uh I went into a lot of depth in integer positional
33:19
encodings, binary positional encodings, sinosidal and now rotary. I believe this understanding is very important because
33:25
if you directly take a look at this paper, it's very difficult to understand what they have exactly done if you don't
33:31
know rope or rotary positional encoding. And to know rotary positional encoding, you need to appreciate sinosidal
33:36
positional encoding. And to appreciate sinosidal positional encoding, you need to appreciate integer and binary
33:42
position encoding. So I hope with these set of three lectures on positional encoding, I have given you enough
33:47
information of all of these three types. Uh so thanks everyone. Make notes for
33:53
all of these. So now here are all the notes for positional encodings, right? Uh earlier we started with integer
33:59
position encoding. Then we saw binary position encoding. Then we went to cinosidal position encoding and finally
34:05
now we saw rotary positional encoding. Make notes along with me as you're watching the lectures because concepts
34:11
are getting a bit more advanced now. Uh and now finally we are at a stage where we can see one of the most advanced
34:17
concepts in introduced in deepseek the multi head latent attention mixed with rotary positional encoding. Once you
34:23
understand this concept the rest of what deepseek introduced is slightly at a lower technical level than this but this
34:29
is quite complex and for that we needed these all of these lectures so far. Thanks a lot everyone and I look forward
34:35
to seeing you in the next lecture.

All

From the series






