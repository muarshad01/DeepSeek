
#### DeepSeek Innovation
* __DeepSeek Innovation 1__: Auxiliary Loss Free Load Balancing (Equation 16)
* __DeepSeek Innovation 2__: Shared Experts
* __DeepSeek Innovation 3__: Fine-grained Expert Segmentation

***

* __DeepSeek Innovation 1__: Auxiliary Loss Free Load Balancing (Equation 16)

$$ Training ~Loss + \lambda \times {(Num ~of ~Experts) + \sum f_ip_i}$$

* It remove the 2nd loss term.
  
***

* 20:00

$$b_i = b_i + u \times sign(load ~violation ~error)$$

* 30:00

#### Problems with Traditional MoE
* Knowledge Hybridity
  * Limited Experted
  * Specilized Experts
  * Solution: Have huge number of Experts (DeepSeek)
* Knowledge Redundancy
  * Super Specilized Experts
  * Solution: Have Shared Experts (DeepSeek)

***

* 35:00

#### 2. Shared Experts
* This approch divides the Experts into two groups
1. Experts that process every token, regardless of routing.
2. Experts that handle the tokens selectively, based on the usual routing strategy.

* The main reason to adopt shared experts is to reduce redundancy among experts.

***

* 40:00

#### 3. Fine-grained Expert Segmention
* In fine-grained expert segmention, each large expert FFN (Feed-Forward Network) is split into $$m$$ smaller experts by reducing the hidden dimension of the FFN by a factor of $$\frac{1}{m}$$.

* __Note__: The number of expert parameters and computation costs remain CONSTANT.

***
