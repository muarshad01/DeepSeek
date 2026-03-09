* __Note__: DeepsSeek used MTP gains only during the pre-training process. During inference, DeepSeek just used STP.

***

* 20:00 

$$h_i^k=M_k[RMSNorm(h_i^{k-1}),RMSNorm(Emb(t_{i+k}))]$$

* Eauation 21

***

* 25:00


* During pre-training DeepSeek used all of this MTP models, but during their inference they only use the
main model. So, during inference they did not use the MTP at all. So, they did not do speculative decoding, which we discussed in the previous lecture. During inference they only used the main model for the next token prediction. So it was almost like they used MTP to exploit its densification of training signals, etc., but they did not exploit the faster inference properties of MTP. For inference they stuck to the single token prediction but for pre-training they exploited these advantages which we saw densification of training signals improved data efficiency and better planning these three advantages they exploited because of their multi-token prediction pipeline but for inference as I mentioned they just use the first block over here which they have titled as main model okay so
this is the entire deepseek multi-token prediction pipeline so in the previous lecture we saw the intuition behind what
is MTP and why mult Multi token prediction is actually useful.

***
