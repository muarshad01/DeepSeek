* FFNN - (Expansion, Contraction) network
* MoE - Multiple NNs

***

5:00

* Reduce pre-training time
* Accelerate inference
* Each Transformer block has different experts

* Sparsity
* Routing

* __Step-1__: We start with an input matrix of size (4,8) as below.
* __Step-2__: When we pass the input matrix throught these three experts, we get three output expert matrices.
* The challenge is how to merge these three expert output matrices such that we get a resulting output (4,8) matrix
* __Step-3__: To do the merging, we use a mathod called "load balancing".
* __Sparsity__: How many experts will be active for each token?
* Every token will be sent. to a selected nymber of experts. This is called "load balancing".
* __Step-4__: How much weitage needs to be given to each expert? This is decided by __routing mechanism__.
* Expert Selector Matrix = Input Matrix  X Routing Matrix


***

* 25:00

* __Step-5__: Expert selector weight matrix.

***

* 30:00

* __Step-6__:
* __Step-7__:

***
