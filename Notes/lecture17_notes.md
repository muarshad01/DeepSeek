#### The Absorption Trick

$$
\begin{aligned}
\text{Attention ~Score} &= Q \times K^{T} \\
                        &= XW_Q \times (W_{UK} \times C_{KV})^{T}\\
                        &= XW_Q \times (W_{UK}^{T} \times W_{DKV}^{T} \times X^{T} )\\
                        &=\underbrace{X(W_QW_{UK}^{T})}_{Fixed ~at ~training  ~time}~\underbrace{(XW_{DKV})^{T}}_{This ~needs ~to ~be ~cached.}
\end{aligned}
$$

* $$X(W_QW_{UK}^{T})$$: Absorted Query. Fixed at training time (only compute once).
* $$(XW_{DKV})^{T}$$: This needs to be cached.

***

* 10:00

$$
\begin{aligned}
\text{Attention ~Score} &= Q \times K^{T} \\
                        &= R_{pos}(XW_Q) \times R_{pos}(W_{UK} \times C_{KV})^{T}\\
                        &= R_{pos}(XW_Q) \times R_{pos}(\underbrace{W_{UK}^{T} \times (XW_{DKV})^{T}}_{We ~need ~to  ~recompute ~keys ~for ~all ~tokens}) \\
\end{aligned}
$$

* This will significantly hinder inference efficiency.
* Note: $$W_QW_{UK}^{T}$$ can't be directly absored.

***

* 20:00

#### Decoupled RoPE

$$
\begin{aligned}
\text{Attention ~Score} &= Q \times K^T \\
                        &= [Q_C : Q_R] [K_C:K_R]^{T}\\
                        &= \underbrace{Q_CK_C^T}_{Ratain ~old ~magic ~MLA} + Q_RK_R^T
\end{aligned}
$$

* $$W_Q \to W_{DQ} \to W_{UQ}$$

***

* 25:00

***

* 30:00

|||
|---|---|
|$$d$$||
|$$d_{C}$$||
|$$d_{C}^{/}$$||
|$$d_h^R$$||
|$$n_h$$||
|$$d_h$$||

***

* 40:00

$$Q_C = X(W_{DQ}W_{UQ}W_{UK}^T)$$

***
