#### Code Mixture of Experts (MoE) from Scratch in Python
* [makeMoE: Implement a Sparse Mixture of Experts Language Model from Scratch](https://huggingface.co/blog/AviSoori1x/makemoe-from-scratch)
* [minGPT](https://github.com/karpathy/minGPT)

* __Step 0__: Load packages and import data

***

* 5:00

* T4 GPU
* A100 GPU

* __Step 1__: Define each expert as a NN network

* __Activation Function__: ReLu

***

* 10:00

$$Input ~Matrix \times Routing ~Matrix = Expert ~Selector ~Matrix$$

* __Step 2__: Implement the Router

* __Step 3__: Implement top-k load Balancing

* __Step 4__: Use -inf and apply softmax

***

* 15:00

* __Step 5__: Create a class for Top-K Routing

***

* 20:00

* __Step 6__: Create a class for Noisy Top-K Routing

* __Step 7__: Create the Sparse Matrix of Experts (MoE) module


***

* 25:00


* __Step 8__: Putting together all the building blocks of MoE model

***

* 30:00

* __Step 9__: Code the entire Transformer block: Part 1 (Multi-head attention)

***

* __Step 10__: Code the entire Transformer block: Part 2 (Assemble all layers)


* __Step 11__: Define entire language model architecture 

***

* 35:00

* __Step 12__: Create training and testing data

* __Step 13__: Define LLM Loss

***

* 40:00

* __Step 14__: Define training loop parameters and other hyper parameters

* __Step 15__: Initialize the entire model

* __Step 16__: Run the pre-training loop

* __Step 17__: Inference

***
