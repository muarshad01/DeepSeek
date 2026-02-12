#### [Invideo AI - Video Generator | No Editing Skills Needed](https://invideo.io/?utm_source=google&utm_medium=cpc&utm_campaign=Top16_Search_Brand_Exact_EN&adset_name=InVideo&keyword=invideo&network=g&device=c&utm_term=invideo&utm_content=InVideo&matchtype=e&placement=g&campaign_id=18035330768&adset_id=140632017072&ad_id=616240030555&gad_source=1&gad_campaignid=18035330768&gbraid=0AAAAACqfi_CasH_dati6efWraWC4sWV3x&gclid=Cj0KCQiA7rDMBhCjARIsAGDBuECUPsCNxYndLvG9dbQcl3duZVnCdxZL4bu-YboI7x4VTCTYt6KBjfcaAvMFEALw_wcB)

***

#### Part-1: Innovative Architecture
* Multi-head Latent Attention (MLA)
* Mixture of Experts (MoE)
* Multi-token Prediction (MTP)
* Quantization
* Rotary Positional Encoding (RoPE)

***

#### Multi-head Latent Attention (MLA)
We need to understand the following concepts to truly understand MLA:
* Architecture of LLM
* Self-Attention
* Multi-head Attention
* Key Value (KV) Cache

***

#### Architecture of LLM

| Model | Parameters |
|---|---|
| GPT-3   | 175 Billion|
| GPT-4   | 1 Trillion |
| GPT-4.5 | 5-10 Trillion |

***

* [ChatGPT](https://chatgpt.com/)

#### Transformer Block
1. Layer Norm 1
2. Multi-head Attion --> MLA
3. Dropout
4. Layer Norm 2
5. Feed Forward NN --> MoE
6. Dropout

***

```
A true friend accepts you
```

* __Phase-1__: Isolation
  * The word is isolated from its neighbors
* __Phase-2__: Token ID assignment
  * Book of Token IDs (Vocabulary)
    * Words
    * Sub-words
    * Characters
  * Byte Pair Encoding (BPE) 
* __Phase-3__: Token embedding assignment

* __Phase-4__: Positional embedding assignment (Your position among neighbors matter!)

```
The dog chased another dog
```

* __Phase-5__: Add token embedding to positional embedding.
  * Input embedding = Token embedding + Positional embedding

* __Phase-6__: Now, you're finally ready to onboard the train to the Transformer block.

***

* __Phase-7__: Different compartments of a Transformer block
1. Layer Norm 1
2. Multi-head Attion --> MLA
3. Dropout (Improves generalization performance; prevents over-fitting)
4. Skip connection or shortcut connnection (help gradient to flow through an alternate path; vanishing gradient problem)
5. Layer Norm 2
6. Feed Forward NN (Expansion / Contraction) --> MoE
7. Dropout

***

* __Phase-8__: Going through multiple Transformer block

* __Phase-9__: Normalization layer

* __Phase-10__:

***
