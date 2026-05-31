<p align="center">
  <img src="https://github.com/muarshad01/DeepSeek/blob/main/images/lec24/MTP.png" width="600" height="300" />
</p>

***

|Paper|
|---|
| [DeepSeek-V3 (Jan 2025): Section 2.2 MTP](https://arxiv.org/pdf/2412.19437)|

***

* To enable MTP, some changes need to be made in the LLM architecture.
* These changes start after we come out of the transformer block


<p align="center">
  <img src="https://github.com/muarshad01/DeepSeek/blob/main/images/lec26/hidden_state_0.png" width="600" height="200" />
</p>

* Once the input emdedding matrix passes through the shared transformer trunk, the resulting vector we obtain is called as the **Hidden state $0(z)$.**
* If the input matrix dimension is $(3,8)$ as in the figure above, the dimension of the hidden state is also $(3,8)$.
* Since we need to predict 3 tokens at the same time, we assemble 3 theads:
  * Head 1 will predict the first token
  * Head 2 will predict the second token
  * Head 1 will predict the third token
* MTP sequentially predict additional tokens and keep the complete casual chain at each prediction depth.

***

* 15:00 

#### Merged Matrix
$$h_i^k=M_k[\text{RMSNorm}(h_i^{k-1}); \text{RMSNorm}(\text{Emb}(t_{i+k}))]~~~~~~Equation(21)$$

***

* 25:00

* During pre-training DeepSeek used all of this MTP models, but during their inference they only use the
main model (STP). So, during inference they did not use the MTP at all. So, they did not do __speculative decoding__, which we discussed in the previous lecture.
* During inference they only used the main model (STP) for the next token prediction. So, it was almost like they used MTP to exploit its densification of training signals, etc., but they did not exploit the faster inference properties of MTP.

***
