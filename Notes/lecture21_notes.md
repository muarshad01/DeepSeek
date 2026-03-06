
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



of neural networks now instead of low number of neural networks and the dimension of each neural network here is
40:29
lesser than the let's say the original dimension and this idea of having more
40:35
number of experts is called as fine grained expert segmentation. U again why do we do this?
40:43
Because if the number of experts is small, each experts is for forced to learn a wide variety of knowledge type
40:49
which reduces its specialization. In fine grained expert segmentation,
40:56
uh we can have specialized experts because now that there are more experts, each experts can learn something new and
41:03
that solves the first issue which we started out with that's the knowledge hybridity. If you have limited experts,
41:09
every expert has to have a lot of information, right? But now if you have a huge number of experts maybe every
41:15
expert can be specialized in certain amount of or certain specific knowledge
41:20
and that leads to this super specialized experts. Um so even if you go to mixture
41:27
of experts original paper you'll see that in fine grained expert segmentation. So
41:33
see the first figure on the left hand side is conventional top two routing in mixture of experts. That's conventional
41:39
mixture of experts. When we go to the right first we add the fine grained expert segmentation where we have a huge
41:46
number of experts now and then as we go further to the right then we add a shared expert. So that's the first
41:52
innovation which we saw the shared experts and in the second figure we s we see this fine grand expert
41:59
segmentation. Uh one thing to note is that among all of these three the number of expert parameters and computational
42:06
costs actually remain the same. This this is because although we increase the number of experts the dimensions are
42:12
reduced appropriately so that the number of parameters the number of expert parameters actually remain the
42:19
same. Uh so if you see the mixture of experts paper and you'll see the main
42:25
innovations here have are the fine grain expert segmentation 3.1 and section 3.2
42:31
is the shared expert isolation and it contains everything which I've just shown to you right now. I believe in
42:37
version two and version three also uh these things were introduced. So if you look at version three for example
42:45
uh version three also mentions that they use fine grained experts and uh uh
42:50
shared shared experts but along with this they also introduce the loss preload balancing.
42:57
So if you directly start reading deepseek version 3 it will be very difficult to understand this whole
43:03
section because this entire section on mixture of experts they have compressed it in three paragraphs three or four
43:08
paragraphs. The first paragraph uh second third and four paragraph in
43:14
four paragraph they explain all of these innovations which we have seen right now. But to understand this innovations,
43:20
it was important for you all to understand how mixture of experts actually operates. And that's why we had
43:25
the hands-on mixture of experts lecture the first three lectures. Uh all right. So this is the
43:33
fine grain expert segmentation. And now now let me show you some results which deepse had in their paper to show how
43:40
these innovations actually led to improvement over traditional mixture of
43:45
experts. All right. So as I mentioned the last thing which I want to show you is some results which deepseek had in
43:52
their original MOE or mixture of experts paper and the first major major result
43:57
which I want to show you is over here. So if you look at the right hand side that's the deepseek mixture of experts
44:05
and uh here if you see the activated experts the total experts is one one
44:10
common expert which is the one shared expert and 63 routed experts and out of
44:15
those 63 only seven were activated. So the total number of expert parameters which were activated were only 24
44:23
billion and they compared this to another mixture of experts model which is called G-Shard. So if you look at
44:29
this Gstar 1.5 you'll see that they did not have group group experts because
44:34
that was the main innovation implemented by Deepc. So they just had 16 routed experts out of which two were activated
44:42
and the total number of expert parameters which they have is 2.83 83 billion and the activated expert
44:48
parameters which they had was around.35 billion right and they also
44:56
compared one more dense dense 16 model which did not have which only had 16
45:01
experts which did not have the fine grained expert segmentation and which did not also have any grouped um it was
45:09
a dense model which means all the tokens were routed to 16 experts all the tokens were routed to 16. So the number of
45:16
activated parameters were was 1.89 billion which means that the deepseek
45:22
number of activated expert parameters was around 1.5 times smaller than this
45:28
G-Shard and it was around six six times smaller than this dense 16. So this
45:35
dense there was no sparity implemented here. It was like there were 16 experts
45:40
and every top was routed to all 16. Now in spite of deepseek having so less
45:46
number of expert parameters you'll see that all of in all of the metrics deepseek com performed relatively at the
45:53
same level as these other models. So if you look at this accuracy metric deepsee had 54.8 8 which is equivalent to these
46:00
other models also although these other models had a huge number of parameters this dense 16 had six times more
46:07
parameters whereas in this metric let's say arc challenge accuracy deepseek mixture of experts actually was higher
46:15
than these other models that's incredible right so here this deepsee had fine
46:21
grained expert segmentation so they had much more number of experts u then they
46:27
also had shared experts But their number of expert activated parameters was way lower. So their computational cost was
46:34
way lower and still their accuracies was relatively comparable with the best mixture of experts model out
46:40
there. So here what they have mentioned is deepseek mixture of experts achieves comparable performance with a Gshar
46:48
model containing 1.5 times the expert parameter and computation. In addition, deepseek nearly approaches the
46:54
performance of a dense model with 16 times the number of parameters
47:00
um with 16 times the number of parameters which sets the upper bound fore models. So this table clearly
47:06
proves the effectiveness of the MOE architecture and its innovativeness such
47:12
as the fine grained expert segmentation, the shared experts compared to the traditional mixture of experts
47:18
architecture. Another figure here which I believe is very good is this one figure in which
47:24
you can see many things. First you can see let's say we don't have shared experts. So if you this blue line there
47:32
are different metrics here which have been plotted and the blue line is if we don't have shared experts and all the
47:39
other colors we do have shared experts. We have one shared expert for all the other colors. So now if you check the
47:46
performance on all of these matrix we see that the blue is much lower compared
47:51
to all the other colors right for the first metric blue is lower than all the other colors for the third metric it's
47:58
also lower for the fourth and fifth blue is definitely lower than all the other colors. This actually proves that having
48:04
a shared expert improves the performance of the mixture of expert model by
48:10
reducing the knowledge hybridity problem which we saw on the whiteboard. You remember we saw this knowledge hybridity
48:17
problem. This was only in theory up till now but deepseek actually proved this result by showing that if we have shared
48:24
experts it actually improves the model performance on a wide range of tasks.
48:30
The second thing we can get from this figure is that we can also see the
48:35
effect of fine grained expert segmentation. Right? So this yellow for example this yellow line does not have
48:42
fine grained expert segmentation. It only has 15 experts. Whereas the green and the the green line and the orange
48:50
line have fine grained expert segmentation. They have 31 and 63 routed experts. So you will consistently see
48:57
that the orange line performs the best among all. Right? The orange performs best across all metrics. This is because
49:03
the orange line has 63 routed experts. This again shows the importance of fine grained expert segmentation in solving
49:11
the knowledge redundancy problem. It shows it shows quantitatively that
49:16
having more experts uh having more
49:23
experts means I can have super specialized experts, right? So actually first I mentioned
49:30
that the shared experts solve the knowledge hybridity problem but actually
49:35
the shared experts solve the knowledge redundancy problem. Right? Because if you have shared experts, it means that
49:42
all the other experts don't need to assemble the same knowledge. The other experts can be specialized. So
49:49
this having the shared expert solves the knowledge redundancy problem. Whereas
49:55
having more number of experts, having fine grained expert segmentation solves the knowledge hybridity problem. So if
50:01
you have a limited number of experts, all experts are forced to learn everything. But if you have more number
50:06
of experts, every expert can be super specialized. So this solves the knowledge hybridity problem. Again
50:12
remember having shared experts solves the knowledge redundancy problem and having fine grained expert segmentation
50:19
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













