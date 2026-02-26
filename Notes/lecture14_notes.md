#### Integer and Binary Positional Encodings (PE)


my name is Dr. Raj Dandkar. I graduated with a PhD in machine learning from MIT in 2022 and
0:08
I'm the creator of the build deepseek from scratch series. Before we get started, I want to introduce all of you
0:15
to our sponsor and our partner for this series invido AI. All of you know how
0:20
much we value foundational content building AI models from the nuts and bolts. In Nvidia AI follows a very
0:27
similar principle and philosophy to that of us. Let me show you how. So here's
0:33
the website of Invido AI. With a small engineering team, they have built an
0:38
incredible product in which you can create highquality AI videos from just
0:43
text prompts. So as you can see here, I've mentioned a text prompt. Create a
0:49
hyper realistic video commercial of a premium luxury watch and make it cinematic. With that I click on generate
0:56
a video. Within some time I'm presented with this incredible video which is
1:02
highly realistic. What fascinates me about this video is its attention to detail. Look
1:08
at this. The quality and the texture is just incredible. And all of this has been created from a single text
1:15
prompt. That's the power of Invido's product. The backbone behind the awesome
1:21
video which you just saw is Invido AI's video creation pipeline in which they
1:26
are rethinking video generation and editing from the first principles to experiment and tinker with foundational
1:33
models. They have one of the largest clusters of H100s and H200s in India and
1:38
are also experimenting with B200s. Nvidia AI is the fastest growing
1:44
AI startup in India building for the world and that's why I resonate with them. so much. The good news is that
1:51
they have multiple job openings at the moment. You can join their amazing team. I'm posting more details in the
1:57
description [Music]
2:02
below. Hello everyone and welcome to this lecture in the build deepseek from
2:08
scratch series. Today we are going to start learning about positional
2:14
embeddings. First let me clarify why are we learning about positional embeddings or
2:21
positional encodings in a course about deepseek. So in the previous lecture we
2:28
learned about multi head latent attention and we have started this whole series with the purpose of understanding
2:35
the deepseek architecture in detail and ultimately with the purpose of building the different components of deepseek
2:41
from scratch. So then suddenly why are we starting to learn about positional
2:47
encodings? Let me give you a little bit of a background around that. So here's
2:52
the deepseek version two or deepseek v2 paper which came out in June
2:57
20124. If you look at the multi head latent attention section over here they
3:02
first start out with the simplified multi head latent attention which is what we have seen in the previous
3:07
lecture. But after this point in section 2.1.3 they introduced something which is
3:14
called as the decoupled rotary position embedding. What they did in this
3:19
decoupled rotary position embedding. Let me go back to that
3:25
section. Yeah. What they did in this decoupled rotary position embedding is that they combined multi head latent
3:33
attention with something called the rotary position embedding. And that leads to a much more powerful version of
3:39
the multi head latent attention. In the latent attention mechanism which we saw, we did not include the positional
3:46
embedding, the rotary positional embedding. So to understand this advanced multi head latent attention
3:53
mechanism, we really need to understand what rotary positional embedding means.
3:59
And if you see the deepseek version 3 paper which came out in 2025 and which
4:06
ultimately led to this whole deepseek revolution it led to deepseek R1
4:12
etc. You'll see that here they directly start with this multi head latent
4:18
attention which by default uses rotary positional encoding or rotary positional
4:23
embedding. So in this lecture I'm going to use embedding and encoding interchangeably but it means the same
4:29
thing. So the reason we are having this two to three lectures now on positional
4:35
embeddings or positional encodings is that we ultimately want to understand what rotary positional encodings are and
4:42
then we'll understand how the multi head latent attention is mixed with rotary positional embedding to create a much
4:49
more advanced version of the latent attention mechanism. So the way I'm going to divide this is
4:56
that in today's lecture I'm going to introduce you to what positional embeddings are. So today's lecture I'm
5:03
going to introduce you to what positional embeddings are. And then we are going to look at two types of
5:08
positional embeddings. Today we are going to look at integer positional embeddings and binary positional embeddings. That's the purpose of
5:15


***


today's lecture. In the next lecture we are going to look at sinosidal positional embeddings.
5:22
These sinosidal positional embeddings were introduced in the attention
5:27
um attention is all you need paper. So we are going to look at these
5:33
these uh embeddings or these encodings next. If you scroll down in the paper, you'll see that these are the sinosidal
5:40
positional embeddings. And then in the lecture after that, we are finally going to look at rotary positional
5:47
embeddings which will help us understand how this rotary positional embeddings are mixed with latent attention. The
5:54
reason I want to go through this sequential procedure is again I want you to know every single thing from scratch.
6:00
So if we directly start with rotary positional embeddings, some understanding, some concepts will be
6:06
lost. I want to take you through how these advancements were actually discovered. So I'll go from step number
6:12
one in today's lecture to step number two and then to step number three. All right. So let's get started
6:19
with today's lecture in which we will cover these three things. Introduction to positional embeddings, integer
6:24
positional embeddings and binary positional embeddings. All right. So the first
6:29
thing is what are positional embeddings and why do we need positional embedding
6:35
or positional encoding in the first place. The main reason is that the position at which a word comes in a
6:43
sentence is very important to the context of the sentence. Let me clarify this. So if
6:50
there is a sentence called the dog chased another dog. Okay. And there are
6:55
two dogs here. Right? Let's say I do not consider positional embedding or
7:01
positional encoding at all. What does it mean I do not consider positional embedding? Well, it means that usually
7:07
in the input block of the transformer architecture, let me mark this with a different color. In this input block of
7:14
a transformer architecture, when a given input text comes in, it's first tokenized. It's converted into token
7:20
embeddings. we add positional embeddings to the token embeddings and that leads to something called as the input
7:26
embeddings which is then passed to the transformer block. So let's say we don't have this positional embedding layer at
7:33
all and we just have the token embeddings and let's say if I pass in
7:38
now this sentence the dog chased another dog I have passed in through the input block
7:45
what will happen is that both of these dogs will get converted into token embeddings right so the token embeddings
7:52
are same for these two because they are the same words so the token embedding for the first dog is this the token
7:58
embedding vector for the second block is this. Both of these are completely identical vectors. And now both of these vectors
8:05
will pass just like that to the transformer block without adding any kind of a positional embedding. This
8:11
means that for these two tokens, the input to the transformer block is
8:17
exactly the same. Which further means that when we come out of the attention
8:22
block and when we get the context vector. So we get the context vector for the first dog and we get the context
8:29
vector for the second dog. Dog number two and this is dog number one. These
8:35
two context vectors will be exactly the same because now the input to the transformer block for these two dogs is
8:42
exactly the same. So they'll go through the exact same operations in the transformer block and the context vector
8:48
for these two dogs will be exactly the same. And this is not what I wanted at
8:53
all because I wanted my transformer block to actually capture the fact that these two are different
8:59
dogs. I don't want my transformer to think that these are the same dogs. I want the context vector for this dog and
9:06
this dog to be completely different. Right? That's why position is
9:12
very important. If we don't use positional encoding, the attention mechanism output for both these dogs
9:20
will be exactly the same. That's not good. After coming out of the attention
9:25
block, we want the model to understand the context of the sentence. We want the model to understand that the first dog
9:33
who is chasing is different from the second dog who is being chased. Right?
9:38
So before we enter the transformer block, we want to create some distinction between this dog and this
9:45
dog. And the way we do that is by adding another vector to the token embedding vector and that's this positional
9:52
embedding vector right over here. So there is a very nice tutorial on hugging
9:58
face. What they do is that they take the same example the dog chased another dog and they pass it to through a
10:04
pre-trained llama 3.2 architecture. What they do after passing this um this
10:10
sentence through this architecture is that they let this sentence go through the attention block without positional
10:16
embedding. So they only take the token embedding. They don't add positional embeddings and then they compute the
10:21
output vector or the context vector of both of these dog one and dog two and then they print is it identical and they
10:28
get that the context vector for both of these dogs is identical. And here they conclude
10:34



***



that without any positional information, the output of a self attention operation
10:39
is identical for the same token. That's not good at all. We want the output of
10:44
the attention mechanism to be different for different tokens, right? Which are in different positions because these
10:50
dogs mean completely different things. That's why position embedding or positional encoding is very very
10:57
important. So another example of this is that
11:04
the dog chased the
11:10
ball. It could not catch it. Now see what's happening
11:19
here. If we don't use positional encoding, this it would be the same as this it, right? But that's not good
11:26
because this it actually refers to the dog and this second it refers to the
11:31
ball. So I want this I want the transformer block to know that the input
11:37
embedding for this it and this it is different. So they can refer to different things. That's why it's very
11:43
important to somehow add the positional embedding information to the token
11:48
embedding information. We need to let the transformer know at which position that given token comes into the picture.
11:56
Right? So that's the first takeaway from today's lecture that positional embeddings are very important along with
12:03
token embeddings. If we don't have positional embeddings, two words which look the same, if they are in different
12:09
positions, they will be treated the same way by the transformer, which is not good because words which look the same
12:16
can refer to completely different things. Great. Then let's start thinking about
12:22
how do we do positional embedding, right? How do we do positional encoding? One more goal of mine for today's
12:29
lecture is for you to start thinking from first principles. So, pause this video for a moment and think that okay,
12:36
let's take this same example. The dog chased another dog and you have got the token embedding and you somehow want to
12:42
add a positional embedding vector to each of these token embedding vectors. How will you encode the information
12:48
about the position? Think about this for a moment. You can pause this video also.
12:54
All right. So, the simplest way to think about this is that okay, if I want to
13:00
encode the position information, let me just do this. Let me take the position number of this token in the sentence.
13:06
So, this come this dog comes at position number two and this dog comes at position number five. That's it. I'll
13:13
just use these positions and add it to my input embedding vector. So the way
13:18
I'll do that is that if the embedding size is equal to 8, let's say, and if
13:24
dog comes at a position number 200, I'll just use 200 and then I'll create eight
13:31
copies of it. So the positional encoding vector for dog will be 200 200 200 200
13:37
200 200 200 200 and then I'll just add it
13:42
um I'll just add it to the token embedding vector what will be the problem with
13:49
this first let me explain to you how this will work in practice if we take the sentence the dog chased another dog
13:55
right let's take a look at the first dog in this sentence so in the whole input
14:00
context if If the first dog comes at position number 200, we'll assign it a value of 200. So this will be 200.
14:08
Chased will be 201. Another will be 202. And the second dog will be
14:14
203. Right? So the first dog will so every value here will be now repeated
14:19
eight times because that's the size of the token embedding vector and we need to add both of these vectors. So the
14:26
first dog will be 200 repeated eight times and the second dog will be 203
14:31
repeated eight times. What's the issue with this approach? What's the problem with this
14:38
approach? We are achieving what we set out to achieve, right? What I wanted to achieve was that I wanted the
14:43
transformer to know that the input embedding vector for this dog and this dog is different. That's what we are
14:49
achieving. For the first dog, we are adding a vector of 200s. For the second dog, we are adding a vector of 200. and
14:55
threes. So the input embedding vector which is the sum of the token embedding plus position embedding will be
15:01
different for the two dogs. So if we are achieving what we what our original aim was, what's the main problem with
15:07
integer positional encoding? Well, the main problem with
15:13
integer positional encoding is the magnitude of the values itself. So if
15:18
you look at token embedding values and the way token embedding values are usually initialized in large language
15:24
model architectures is that these values are usually clustered around
15:31
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


