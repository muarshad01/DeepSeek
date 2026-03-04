* There are two major issue with MoE models, which we are going to look at in Steps 8 & 9.

#### __Step 8__: Auxiliary loss version
* In MoE models, the routing mechanism selects a subset of experts for each input.
* If some experts are chosen too often while others are underutilized,  it leads to inefficient learning and potential performance bottleneck.
* An auxiliary loss term is added to main training loss to penelize imbalanced expert selection, pushing the routing functin toward a more uniform distribution.
* To calcuate the auxiliary loss, we first start with expert selector weight matrix, which consistes of the experts assigned to every token and the probabilities assigned to every expert.

***

* 10:00

* __Expert Importance__
* Balanced MoE model: All experts should have equal importance

* We should penalize the MoE model if the expert importance score have a lot of variation. This would mean that some experts are too important than others.
* The mathematical quantity, which will help us do this is __Coefficient of variation (CV)__.

$$CV = \frac{Standard ~Devivation (\sigma))}{mean(\mu)}$$

$$Auxiliary ~Loss = \lambda \times (CV)^2$$

* Addition of this loss term will ensure that the expert importance is uniformly distributed among different experts.

***

* 15:00

* __Step 9__:  Load Balancing
* While the previously disccused importance loss is useful, assigning equal importance to experts doesn't necessarity lead to uniform token routing.

* If the tokens sent to each expert is non-uniform, it can lead to high memory and a reduced MoE performance.

***

* 20:00
* P1
* P2
* P3
* The next step is to calculate the fraction to tokens dispatched to each expert.

* $$Load ~Balancing ~Loss = Scaling ~Factor \times ~Number ~of ~Experts \times \sum_{i=1}^{num-experts}f_ip_i$$

***

* 30:00

* In other words, minimizing this loss encourages:
* Experts with high importance (p_i) to potentially handle more tokens, this increasing f_i.
* Experts with low importance to handle proportionally fewer tokens, decreasing mismatch.

* This alignment reduces overall imbalance between assigned importance (probability) and actual routing (disptach).



* By minimizing the auxiliary loss, you mathematically enforce the model to distribute tokens proportionally to how much each expert is valued or "trusted", resulting in more efficient and balanced use of the MoE architecture.
*


* __Step 10__:  Capacity Factor
* Problem: Expert Imbalance

$$Expert ~Capacity = \frac{Tokens ~per ~Batch}{Number ~of ~Experts} \times Capatiy ~Factor$$

***

* 35:00

$$Tokens ~per ~batch = Batch ~size \times Sequence ~Length \times Top-k$$
* Top-k: Number of experts choses for each token (load balancing)

* Capacity factor is a scaling factor (typically > 1) to allow some flexibility.

* If capacity factor = 1, each expert gets an even split of tokens.
* If capacity factor > 1, each expert can handle more than its fair share, allowing load balancing.
* If capacity factor < 1, some tokens may get dropped if all experts are full.

* When expert capacity is excedded, excess tokens are eiter dropped or handles by another expert (depending on the implementation).

***

* 40:00

* Unlike dense models where all the parameters are used for every input, MOE only activates a subset of
experts (e.g., 1 or 2 per token). This reduces the number of active parameters per forward pass, leading to:
1. Lower FLOPS
2. Reduced memory footprint, allowing for larger models without too much increase in costs.

#### DeepSeek Innovation
* __DeepSeek Innovation 1__: Auxiliary Loss Free Load Balancing
* __DeepSeek Innovation 2__: Shared Experts
* __DeepSeek Innovation 3__: Fine-grained Expert Segmentation

***
