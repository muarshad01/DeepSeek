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

factor. Okay. So now once you learn all of this ultimately it's important to
40:11
understand the key advantages of mixture of experts. So if you have 32 64 128
40:17
experts you you get a speed up of five to seven times. That's normal to see with mixture of experts. You get the
40:23
training time speed up of a huge amount if you use mixture of experts versus if you do not use mixture of
40:30
experts. Uh so unlike dense models where all the parameters are used for every input, MOE only activates a subset of
40:37
experts and uh what this does is that it reduces the number of active parameters per forward pass and that leads to lower
40:45
operations per step and it also leads to reduced memory footprint allowing for larger models without too much increase
40:52
in costs. So this one figure over here sums up the advantages of mixture of experts. Right? It gives a huge amount
40:58
of speed up as compared to if we do not use a mixture of experts model. In the previous lecture step one
41:05
to step seven that covers the main mechanism of mixture of experts. But without understanding step uh step
41:11
number eight, step number nine uh and the expert capacity which was
41:17
step number 10, it is impossible to understand the deepseek innovations which came because if you look at the
41:23
deepseek innovations, the major innovations is something called auxiliary loss loss load balancing and
41:29
you would not be able to understand this innovation if you don't understand what load balancing is.
41:35
uh so that's why we have to cover this lecture where I go through all of these concepts and then we are also going to
41:41
look at other deepseek innovations like uh shared experts and the final
41:46
innovation is fine grain expert segmentation uh we are going to look at all of these
41:52
deepseek innovations in the coming lectures so stay tuned make notes while I'm explaining these lectures so that
41:58
you understand these lectures in a much in a much much better manner uh it is
42:04
very hard to find videos which talk about these things in a lot of detail because these are not easy concepts such
42:10
as uh auxiliary loss or load balancing but I want to show you everything from
42:15
scratch and that's why I'm showing everything like this on a whiteboard. Thanks everyone. In the next lecture
42:22
we'll be embarking we'll be embarking on a journey to understand the deepseek
42:27
innovations in the mixture of experts modeling. So thanks a lot and I look forward to
42:33
seeing all of you in the next lecture.


















