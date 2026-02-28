* 15:00

* Note that to include positional information, we don't add any vector to the query vector. 
* We effectively rotate parts of the original query vector and hence maintain its magnitude. Thus avoiding the token embedding polluting issue which we saw with sinosidal positional encoding.

***

* 20:00

* [RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE) - (2023)](https://arxiv.org/abs/2104.09864)

***

* 25:00

* Higher index components ensure that even with large position differences the relationship is preserved.

```
Einstein developed the theory of relativity. This breakthrough reshaped physics.
```

* "This breakthrough" refers to "the theory of relativity", several words earlier.
* So, higher index oscillations capture these long-range context dependencies.

***

