<p align="center">
  <img src="https://github.com/muarshad01/DeepSeek/blob/main/images/lec24/MTP.png" width="500" height="300" />
</p>

***

* 20:00 

$$h_i^k=M_k[RMSNorm(h_i^{k-1}),RMSNorm(Emb(t_{i+k}))]$$

* Eauation 21

***

* 25:00

* During pre-training DeepSeek used all of this MTP models, but during their inference they only use the
main model (STP). So, during inference they did not use the MTP at all. So, they did not do __speculative decoding__, which we discussed in the previous lecture.
* During inference they only used the main model (STP) for the next token prediction. So, it was almost like they used MTP to exploit its densification of training signals, etc., but they did not exploit the faster inference properties of MTP.

***
