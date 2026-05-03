#### Multi-Head Latent Attention (MLA)

* How DeepSeek Changed Attention?

1. Multi-Head Attention (MHA)
2. Multi-Query Attention (MQA)
3. Grouped-Query Attention (GQA)
4. Multi-Head Latent Attention (MLA) - Compressed Latent KV-Cache

***

* 5:00

* __Takeaway-1__: We seem to be doing a lot of repeated calculations.

***

* 15:00

* __Main Observation__: To get the logits vector for `bright`, we only need the context vector for `bright`.
* __3__: What do we really need to predict the next token?
* __4__: What if we store/cache the keys and valus matrices during inference?
* __5__: We need to cache Kay ($K$) and Value ($V$) matrices. This is called a KV-cache. We don't need to store Queries ($Q$) matrix.
* __6__: KV-cache advantages.
  * $\text{Computation Cost} = \mathcal{O}(\text{num tokens})$

***

```
The next day is
```
* X(4,8)
* $Q(4,4) = X(4,8) \times W_Q(8,4)$
* $K(4,4) = X(4,8) \times W_K(8,4)$
* $V(4,4) = X(4,8) \times W_V(8,4)$
* $A^{score}(4,4) = Q(4,4) \times K^T(4,4)$
* $A^{weight}(4,4) = Q(4,4) \times K^T(4,4)$
* $Z(4,4) = A(4,4) \times V(4,4)$

***

#### New token $X_{bright}(1,8)$
* X(5,8)
* $Q(5,4) = X(5,8) \times W_Q(8,4)$
* $K(5,4) = X(5,8) \times W_K(8,4)$
* $V(5,4) = X(5,8) \times W_V(8,4)$
* $A^{score}(5,5) = Q(5,4) \times K^T(4,5)$
* $A^{weight}(5,5) = Q(5,4) \times K^T(4,5)$
* $Z(5,4) = A(5,5) \times V(5,4)$

***

* $Q_{bright}(1,4) = X_{bright}(1,8) \times W_Q(8,4)$
* $K_{bright}(1,4) = X_{bright}(1,8) \times W_K(8,4)$
* $V_{bright}(1,4) = X_{bright}(1,8) \times W_V(8,4)$
* We just need to make the above three ($\uparrow$) computations for every new inference!
* $A_{bright}^{score}(1,5) = Q_{bright}(1,4) \times K_{bright}^T(4,5)$
* $A_{bright}^{weight}(1,5)$
* $Z_{bright}(1,4) = A_{bright}^{weight}(1,5) \times V(5,4)$
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
                            &=\underbrace{X(W_QW_{uK}^{T})}_{\text{Fixed at training time.}}~\underbrace{(XW_{dKV})^{T}}_{\text{This needs to be cached.}}
\end{aligned}
$$

#### Absorted Query
* $X(W_QW_{uK}^{T})$ 
* Fixed at training time (Only compute once!).

$$
\begin{aligned}
\text{Context Matrix} &=  \text{A} \times V \\
                             &= (QK^{T})(C_{KV} \times W_{uV}) \\
                             &= (QK^{T})(X \times W_{dKV} \times W_{uV}) \\
                             &= (QK^{T})(X \times W_{dKV} \times W_{uV}) \times W_{0}: \text{Logits Matix}\\
                             &= (QK^{T})(X \times W_{dKV})(W_{uV} \times W_{0})\\
                             &= (\text{Attention Scores})(\text{Cached})(\text{Fixed at traing - Only commputed Once})\\
\end{aligned}
$$

* $C_{KV}=XW_{DKV}$: We only cache this and share across ALL attention heads.
* Unlike MQA, the shared latent matrix is projected back into Keys ($C_{KV} \times W_{uK}$) and Values ($C_{KV} \times W_{uV}$) matrices, which are different for each attentionhead.
* Thus, all heads have different K,V values. This solves the performance proboem of MQA.

***

* 45:00

#### Example

#### Letent KV-Cache

* $C_{KV}(4,4) = X(4,8) \times W_{dKV}(8,4)$

* __15__: So, what happens when a new token comes in?
* First, we compute the queries project into latent space.

* __1__: New arriving vector $X^{bright}(1,8)$

$$
\begin{aligned}
   Q^{bright}(1,4) &= X^{bright}(1,8) \times W_Q(8,4)\\
\end{aligned}
$$

* __2__: Compute $C_{KV}^{bright}$ Vector

$$
\begin{aligned}
   C_{KV}^{bright}(1,4) &= X_{bright}(1,8) \times W_{dKV}(8,4)\\
\end{aligned}
$$

* __3__: Update Latent Cache: $C_{KV}$
  * Updated $C_{KV}(5,4)$

* __4__: Compute $K$ and $V$

$$
\begin{aligned}
   K(5,4) &= C_{KV}(5,4) \times W_{uK}(4,4)\\
   V(5,4) &= C_{KV}(5,4) \times W_{uV}(4,4)\\
\end{aligned}
$$

* __5__: Attention Score (A)
  * $A^{bright}(1,5) = Q^{bright}(1,4) \times K^T(4,5)$

* __5__: Attention Weight (Z)
  * $Z^{bright}(1,4) = A^{bright}(1,5) \times V(5,4)$

***

* 50:00

* __16__: Does this solves the two problems we started with?
* Can we get the best of both worlds?
1. Low cache size
2. Good language model performance

* $h : d_{head}$ : hidden dim or head dimension
* $n_{heads}$ :  number of heads

$$
\begin{aligned}
   MHA &:  l \times b \times s \times \underbrace{n_{heads} \times h}_{\text{embedding dim}} \times 2 \times 2\\
   MQA &: l \times b \times s \times \underbrace{1\times h}_{\text{embedding dim}} \times 2 \times 2\\
   GQA &: l \times b \times s \times \underbrace{g \times h}_{\text{embedding dim}} \times 2 \times 2 \\
   MLA &: l \times b \times s \times d_{latent} \times 1 \times 2 \times \\
\end{aligned}
$$

$$
\begin{aligned}
  \text{Reduction }        &= \bigg(\frac{d_{latent}}{2 \times n_{heads} \times h}\bigg) \\
  \text{For DeepSeek}      &= \frac{2 \times 128 \times 128}{576} \approx 64\\
\end{aligned}
$$

* For DeeSeek, memory reduction from 400 GB to $\frac{400}{64} = 6.25 GB$

***

* $W_{dKV}(embedding-dim, letent-dim)$
* DeepSeek embedding dimention = $n_{heads} \times d_{head} = 128 \times 128$
* Latent Space dimension = 512
* Insted of two K and V matrices, we only have on $C_{KV}$. That is a reduction by factor of 2
* Reduction along dimension = $\frac{128 \times 128}{512}$
* Total reduction = $\frac{2 \times 128 \ times 128}{512} = 64$


***
