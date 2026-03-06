
#### DeepSeek Innovation
* __DeepSeek Innovation 1__: Auxiliary Loss Free Load Balancing
* __DeepSeek Innovation 2__: Shared Experts
* __DeepSeek Innovation 3__: Fine-grained Expert Segmentation

***

* 5:00

***

* 10:00

***

* 15:00

* __DeepSeek Innovation 1__: Auxiliary Loss Free Load Balancing (Equation 16)

***

* 20:00

$$b_i = b_i + u \times sign(load ~violation ~error)$$

***

* 25:00

***

* 30:00

#### Problems with Traditional MoE
* Knowledge Hybridity
  * Limited Experted
  * Specilized Experts
  * Solution: Have huge number of Experts (DeepSeek)
* Knowledge Redundancy
  * Super Specilized Experts
  * Solution: Have shared Experts (DeepSeek)

***

* 35:00

#### Shared Experts
* This approch divides the Experts into two groups
1. Experts that process every token, regardless of routing.
2. Experts that handle the tokens selectively, based on the usual routing strategy.

* The main reason to adopt shared experts is to reduce redundancy among experts.

***

* 40:00

#### Fine-grained Expert Segmention
* In fine-grained expert segmention, each large expert FFN (Feed-Forward Network) is split into $$m$$ smaller experts by reducing the hidden dimension of the FFN by a factor of $$\frac{1}{m}$$.

* __Note__: The number of expert parameters and computation costs remain CONSTANT.

***

* 45:00




***


solves the knowledge hybridity problem. So what they showed here quantitatively is that this orange line has fine
50:26
grained expert segmentation. Right? That solves the knowledge hybridity problem.
50:33
uh and this yellow line compared to the blue line we have shared experts. So
50:38
that solves the knowledge redundancy problem. So again to repeat it shared
50:43
expert solves the knowledge redundancy problem and fine grain expert segmentation solve the knowledge
50:49
hybridity problem. Again in this plot you can see across all the different metrics the
50:56
orange line is the highest and the orange line has shared expert plus fine grain expert segmentation. That proves
51:01
that having shared expert plus fine grain expert segmentation solves both
51:07
the knowledge the redundancy and knowledge hybridity problem and it leads to better performance across all of
51:13
these metrics which have been considered over here. They have several other results over
51:18
here which improve which show that how deepsek mixture of experts achieve super specialization using these
51:26
innovations. So that brings us to the end of today's lecture in which we have comprehensively seen the three main
51:32
innovations which power the deepseek mixture of experts. The first is uh
51:37
auxiliary loss free load balancing. The second is shared experts and the third
51:44
is finerained expert segmentation. These innovation two and three solve the problem of knowledge hybridity and
51:51
knowledge redundancy. Whereas the innovation number one which is the auxiliary loss preload balancing
51:57
reduces. We don't need to have this second loss term now which means that we can solely focus on reducing my training
52:04
loss and by dynamically adjusting this bias term. Now by dynamically adjusting
52:10
this bias term we can make sure that all the experts have equal amount of load.
52:16
So if some expert is underloaded we increase the bias. If some expert is overloaded we reduce the bias which
52:22
ultimately makes sure that the probability the router assigns to every expert is more or less uniform across
52:28
the experts. So what deepseek did is that they took they built upon something
52:34
which was already existing. They did not invent the mixture of experts architecture but they were smart enough
52:40
to notice the limitations in mixture of experts. Uh like there was a limitation in this loss term right when we added
52:46
this auxiliary loss term. When we added this uh this one this auxiliary loss
52:53
term uh there was a trade-off with the language performance. So they mitigated that trade-off by getting rid of this
52:59
term. Then they also added these two uh shared experts and
53:05
uh fine grain expert segmentation to solve the knowledge hybridity and knowledge redundancy problem. Everyone
53:11
had noticed these problems but deepseek was one of the first ones to take steps towards mitigated mitigating it and
53:18
that's the beauty of everything which they have done. They showed that a group of people if they came together even if
53:23
the resources are not that high we can still achieve incredible and innovative things and that's one of the main
53:30
inspiration why I'm making this series. Thanks a lot everyone. There are lots more lot more interesting
53:37
uh and advanced concepts to follow. So please stay tuned and make notes so that you'll understand and follow all. Thanks
53:44
everyone and I look forward to seeing you in the next lecture.















