#### The Absorption Trick

$$
\begin{aligned}
\text{Attention ~Score} &= Q \times K^{T} \\
                        &= XW_Q \times (W_{UK} \times C_{KV})^{T}\\
                        &= XW_Q \times (W_{UK}^{T} \times W_{DKV}^{T} \times X^{T} )\\
                        &=\underbrace{X(W_QW_{UK}^{T})}_{Fixed ~at ~training  ~time}~\underbrace{(XW_{DKV})^{T}}_{This ~needs ~to ~be ~cached.}
\end{aligned}
$$

* $$X(W_QW_{UK}^{T})$$: Absorted Query. Fixed at training time (only compute once).
* $$(XW_{DKV})^{T}$$: This needs to be cached.

***

* 10:00

$$
\begin{aligned}
\text{Attention ~Score} &= Q \times K^{T} \\
                        &= R_{pos}(XW_Q) \times R_{pos}(W_{UK} \times C_{KV})^{T}\\
                        &= R_{pos}(XW_Q) \times R_{pos}(\underbrace{W_{UK}^{T} \times (XW_{DKV})^{T}}_{We ~need ~to  ~recompute ~keys ~for ~all ~tokens}) \\
\end{aligned}
$$

* This will significantly hinder inference efficiency.
* Note: $$W_QW_{UK}^{T}$$ can't be directly absored.

***

* 15:00

***

* 20:00

#### Decoupled RoPE

$$
\begin{aligned}
\text{Attention ~Score} &= Q \times K^T \\
                        &= [Q_C : Q_R] [K_C:K_R]^{T}\\
                        &= \underbrace{Q_CK_C^T}_{Ratain ~old ~magic ~MLA} + Q_RK_R^T
\end{aligned}
$$

* $$W_Q \to W_{DQ} \to W_{UQ}$$

***

* 25:00


***

* 30:00

|||
|---|---|
|$$d$$||
|$$d_{C}$$||
|$$d_{C}^{/}$$||
|$$d_h^R$$||
|$$n_h$$||
|$$d_h$$||

***

* 40:00

$$Q_C = X(W_{DQ}W_{UQ}W_{UK}^T)$$

***

* 50:00

***

* 55:00

***

1:00:00


shown over here u by 57 times. So this is 9x2 HL. Where does this 9x2 come
1:00:27
from? So DC is what we are calling DL over here that's 4 * H or 4 * DH and DHR
1:00:35
is DHX2. So overall this comes to 9 DH2. So what you can see over here is
1:00:42
the capability right. uh a multi head attention is a strong performance whereas GQA and MQA don't have good
1:00:49
performance whereas MLA so when they say hours it's MLA plus rope that has a
1:00:55
great model performance plus at the same time they reduce the KV cache by a huge
1:01:00
amount compared to the multi head attention so with MLA plus rope we truly get the best of both worlds we get the
1:01:06
capability plus at the same time we get the KV cache memory reduce uh reduction
1:01:12
I hope all these This formula makes sense to you. Now once you visually understand what is happening on the whiteboard in these computations, you
1:01:19
will know why there is DL plus DHR here which needs to be cached. So Deepseek had this graph where
1:01:26
they actually compared MHA and MLA. Forget about the MO part for now. That
1:01:32
is the mixture of experts. We are going to come to that later. For now just look at the MHA and MLA part. you'll see that
1:01:39
the MLA performance is mostly equivalent or better than the M uh than the MHA and
1:01:47
that's amazing right we actually get a better performance with latent attention while at the same time we save the cash
1:01:54
so here you see KV cache per token in MLA is 34.6K 6k and here it's 860. So
1:02:02
here there is a factor of 25 or 30 and here again there is a factor of 6 or 7.
1:02:08
So we save the memory plus at the same time you get great performance. That's the beauty of the latent attention with
1:02:15
rotary positional encodings. And the reason I believe deepseek added rotary positional encodings is to get the
1:02:21
better performance values which we see over here in all these benchmarks. But they truly achieved the best of both
1:02:27
worlds. they reduced the memory size and got a good performance. In this one lecture I uh we
1:02:33
have covered everything which is present in the deepseek version two and version three paper on multi head latent
1:02:38
attention. I believe I have not left out anything. Now you can truly understand this entire entire diagram. I already
1:02:46
explained you this diagram previously but now you can see these you can understand the shaded portions also. You
1:02:51
only need to cache the KR that two only for one head and the
1:02:57
latenc that's why it's shown with a dash over here and that's why it's shown cache during inference. So you will
1:03:03
you'll understand the schematic you'll understand all of these equations 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
1:03:13
19. So the this I believe until here when you start reading the deepseek paper the architecture part of it is
1:03:20
extremely important but it's also very difficult for people to just read through it you need to make notes and
1:03:25
even when you make notes you need to go deeper into it I hope this lecture serves as a key facilitator for you to
1:03:32
understand everything in the deepseek paper just like I have done with uh latent attention my next target is going
1:03:38
to be a mixture of experts so the mixture of experts is one of another
1:03:43
incredible innovations in the deepseek paper. We are going to go through that and after that we are going to go
1:03:48
through u multi-token prediction. Thanks a lot everyone and this was one
1:03:53
foundational lecture in building and understanding deepseek from scratch. It took a very long time for me to prepare
1:04:00
this lecture but I hope it was worth it and I hope all of you have really understood how latent attention plus
1:04:06
rotary positional encoding was implemented by deepseek. Thanks a lot everyone and I look forward to seeing
1:04:12
you in the next lecture.






















