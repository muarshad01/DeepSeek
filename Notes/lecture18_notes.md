#### Mixture of Experts

***

| Paper ||
|---|---|
| [(1) 1991 paper on Mixture of Experts](https://www.cs.toronto.edu/~fritz/absps/jjnh91.pdf)||
| [(2) ST-MoE paper (2022)](https://arxiv.org/pdf/2202.08906)||
| [(3) DeepSeek MoE (Jan 2024)](https://arxiv.org/pdf/2401.06066)||
| [(3) DeepSeek V2 (June 2024)](https://arxiv.org/pdf/2405.04434)||
| [(4) DeepSeek V3 (Jan 2025)](https://arxiv.org/pdf/2412.19437)||

***

* 10:00

* In a Mixture of Experts (MoE) model, we have multiple neural networks called "experts" in the transformer block.

#### What is the need to add multiple experts?
* There are two main advantages:
1. Adding multiple experts allow models to be pre-trained with far less compute compared to a dense model (without experts).
2. Allows much faster inference compared to a dense model (without experts)


* In a dense model, every input token passes through all the parameters(i.e., all layers and neurons).
* In contrast, a Mixture of Experts (MoE) model has multipe "experts" (think of them as sub-networks or feedforward layers), but only a small subset (e.g.,2 out of 64) are activated for any given input.

### Sparsity
* Sparsity is one of the main ideas behind MoE

***

* 20:00

* [The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)

```
What is 1 + 1 ?
```

***
