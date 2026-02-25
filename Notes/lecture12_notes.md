#### Multi-Head Latent Attention (MLA)

* How DeepSeek Changed Attention?

* [DeepSeek-V2: A Strong, Economical, and Efficient
Mixture-of-Experts Language Model - 2024](https://arxiv.org/pdf/2405.04434)

1. Key-Value (KV) cache
2. Multi-Head Attention (MHA)
3. Multi-Query Attention (MQA)
4. Grouped-Query Attention (GQA)
5. Multi-Head Latent Attention (MLA)

***

* 5:00

#### Key-Value Cache
* __Takeaway-1__: We seem to be doing a lot of repeated calculations.

***

* 15:00

* __To get the logits vector for "bright," we only need the context vector for "bright".__

* __4__: What if we store/cache the keys and valus matrices during inference?
* __5__: We need to cache Kay and Value matrices. This is called a K-V cache. We don't need to store Queries (Q) matrix.
* __6__: K-V cache advantages.
  * Computation Cost = O(number of tokens)

***

* 20:00

* __7__: K-V cache disadvantages.
  * Size of K-V cache

* __8__: Solving the K-V cache memory problem

* 30:00

* __9__: Can we get best of both worlds?
  * Low cache size
  * Good language model performance

***

* 35:00

* __10__: What if we don't have to cache the Keys & Values seperately?
  * What if we cache only one matrix?
  * What if this matrix has less dimention than $$n \times h$$?
 
 $$2 \times n \times h \to n_l$$

* __11__: To get this matrix, we start by projecting the input embedding matrix into __latent space!__

$$X: (4,8) \times W_{DKV} (8,4) \to \text{Latent Matrix}: C_{KV}(4,4)$$


***

* 40:00

$$(W_Q,W_K,W_V) \to (W_{Q},W_{UK},W_{UV})$$
* Note: W_Q remains the same but W_K and W_V and projects to W_{UK} and W_{UV}

#### How does adding this latent matrix help?
$$
\begin{aligned}
Q      &= X \times W_Q \\
C_{KV} &= X \times W_{DKV} \\
K      &= C_{KV} \times W_{UK} = X \times W_{DKV} \times W_{UK}\\
C      &= C_{KV} \times W_{UV} = X \times W_{DKV} \times W_{UV}
\end{aligned}
$$

#### The Absorption Trick

$$
\begin{aligned}
\text{Attention ~Score} &= Q \times K^{T} \\
                        &= XW_Q \times (W_{UK}^{T} \times W_{DKV}^{T} \times X^{T} )\\
                        &=X(W_QW_{UK}^{T}) (XW_{DKV})^{T}
\end{aligned}
$$

* $$X(W_QW_{UK}^{T})$$: Absorted Query - Fixed at training time (only compute once).
* $$(XW_{DKV})^{T}$$: This needs to be cached.


$$
\begin{aligned}
\text{Context Vector Matrix} &=  \text{Attention Weights} \times V \\
                             &= (QK^{T})(XW_{DKV}W_{UV}) \\
                             &= (QK^{T})(XW_{DKV}W_{UV})W_{0}: \text{Logits Matix}\\
                             &= (QK^{T})(XW_{DKV})(W_{UV}W_{0})\\
                             &= (\text{Attention Scores})(\text{Cached})(\text{Fixed at traing - Only commputed Once})\\
\end{aligned}
$$

* $$XW_{DKV}$$: We only cache this and share across all attention heads.
* Unlike MQA, the shared latent matrix is projected back into Keys and Values matrices -> $$W_{UK}$$ and $$W_{UV}$$ have weights different for each attention head. Thus, all heads have different K,V values. This solves the performance proboem of MQA.


***

* 45:00

#### Example

#### Letent KC-cache

* $$X(4,8) \times W_{DKV}(8,4) \to KVcache (4,4)$$

* __15__: So, what happens when a new token comes in?
* First, we compute the queries project into latent space.

$$
\begin{aligned}
Q &= X_{bright}(W_Q.W_{UK}^{T})\\
&=X_{bright}(1,8)(8,4)(4,4) \to \text{Absorbed Query vector for bright}(1,4)\\
\end{aligned}
$$

* __2__: Compute KV Vector

$$
\begin{aligned}
    Q &= X_{bright}.W_{DKV}\\
      &= X_{bright}(1,8).(8,4) \to (1,4) \to \text{Append to latent KV cache}\\
\end{aligned}
$$
* Updated KV cache (5,4)

***

* 50:00

* __16__: Does this solves the two problems we started with?
* Can we get the best of both worlds?
1. Low cache size
2. Good language model performance

$$
\begin{aligned}
   MHA &: 2 \times 2 \times l \times b \times s \times n \times h \\
   MQA &: 2 \times 2 \times l \times b \times s \times 1 \times h \\
   GQA &: 2 \times 2 \times l \times b \times s \times g \times h \\
   MLA &: 1 \times 2 \times l \times b \times s \times d_{l} \\
\end{aligned}
$$

$$
\begin{aligned}
\text{Reduction of size} &= \frac{2 \times n \times h}{d_l}\\
\text{For DeepSeek}      &= \frac{2 \times 128 \times 128}{576}\\
                         &\approx 57
\end{aligned}
$$

* Memory reduction from 400 GB to $$\frac{400}{60}=6.6GB$$ for DeepSeek.

***



