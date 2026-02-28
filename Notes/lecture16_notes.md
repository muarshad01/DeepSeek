* 15:00

* Note that to include positional information, we don't add any vector to the query vector. 
* We effectively rotate parts of the original query vector and hence maintain its magnitude. Thus avoiding the token embedding polluting issue which we saw with sinosidal positional encoding.

***

* 20:00

* [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

***

* 25:00

* Higher index components ensure that even with large position differences the relationship is preserved.

```
Einstein developed the theory of relativity. This breakthrough reshaped physics.
```

* "This breakthrough" refers to "the theory of relativity", several words earlier.
* So, higher index oscillations capture these long-range context dependencies.

***

* 30:00

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












