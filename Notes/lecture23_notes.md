#### Multi-Token Prediction Introduction

| Year | Paper |
|---|---|
| Apr 2024 | [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737) |


***

* 10:00

#### Why is MTP useful?
1. Densification of Training Signals
* MTP provides richer and denser training signals than single token prediction. 
* Traditional single token prediction only guides the model to predict a single immediate token.
* MTP, however, instructs the model to simultaneously predict multiple future tokens, generating more informative gradient signals per training example.

2. Improved Data Efficiency
* MTP train models achieved better results on standard benchmarks like HumanEval and MBPP with the same amount of training data, solving about 15% more code problems on average.
* [Mostly Basic Python Problems Dataset (MBPP)](https://github.com/google-research/google-research/blob/master/mbpp/README.md)
* [Evaluating Large Language Models Trained on Code (Jul 2021)](https://arxiv.org/abs/2107.03374)
* [HumanEval](https://github.com/openai/human-eval)

3. Better Planning
* MTP implicitly assigns greater importance to __"choice points"__ - key tokens that significantly influence future outcomes.
* Thus, models learn to prioritize crucial decision-making elements.

4. Higher Inference Speed
* Up to 3x faster inference speed
* [Speed Up LLM Inference with Speculative Decoding](https://medium.com/@genai.works/speed-up-llm-inference-with-speculative-decoding-1fc79701e9d6)

***

