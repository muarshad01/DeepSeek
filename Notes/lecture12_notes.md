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

#### Latent Matirx
```
The next day is
```

$$\text{Latent Matrix} = C_{KV}(4,4) = X(4,8) \times W_{dKV}(8,4)$$
* We cache this Latent Matrix.


***

* 40:00

#### How does adding this latent matrix help?

* __13__: What we do next?

$$
\begin{aligned}
Q(4,4)      &=X(4,8) \times W_Q(8,4)\\
C_{KV}(4,4) &= X(4,8) \times W_{dKV}(8,4) ~~~~\text{Down Projection!}\\
K(4,4)      &= C_{KV} \times W_{uK} = \underbrace{X(4,8) \times W_{dKV}(8,4)}_{C_{KV}} \times W_{uK}(4,4) ~~~\text{Up projection to recover K!}\\
V(4,4)      &= C_{KV} \times W_{uV} = \underbrace{X(4,8) \times W_{dKV}(8,4)}_{C_{KV}} \times W_{uV}(4,4) ~~~\text{Up projection to recover V!}\\
\end{aligned}
$$

* __Note__: $W_Q(8,4)$ remains the same.
* Instead of caching two large matrices, $K$ and $V$, we only cache one smaller, lower dimensional matrix $C_{KV}$. This single matrix becomes our highly-efficient cache.
* When we need the full Keys ($K$) and Values($V$), we can resonstruct them on-the-fly from the compressed latent representation ($C_{KV}$).

***

* __14__: How does adding this Latent matrix help?

#### The Absorption Trick

$$
\begin{aligned}
                     A(4,4) &= Q \times K^{T} \\
                            &= XW_Q \times (W_{uK} \times C_{KV})^{T}\\
                            &= XW_Q \times (W_{uK}^{T} \times W_{dKV}^{T} \times X^{T} )\\
                            &=\underbrace{X(W_QW_{uK}^{T})}_{\text{Fixed at training time\\ Only comput once!}}~\underbrace{(XW_{dKV})^{T}}_{\text{This needs to be cached.}}
\end{aligned}
$$

#### Absorted Query
* $X(W_QW_{uK}^{T})$ 
* Fixed at training time (Only compute once!).

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
* Unlike MQA, the shared latent matrix is projected back into Keys and Values matrices -> $$W_{UK}$$ and $W_{UV}$ have weights different for each attention head. Thus, all heads have different K,V values. This solves the performance proboem of MQA.


***

* 45:00

#### Example

#### Letent KV-Cache

* $$X(4,8) \times W_{DKV}(8,4) \to C_{KV}(4,4)$$

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
      &= X_{bright}(1,8).(8,4) \to (1,4) \to \text{Append to latent K-V Cache}\\
\end{aligned}
$$

* Updated $$C_{KV}(5,4)$$

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










