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

experts. And in the case of load balancing, of course, we take a look at the expert importance through PI, but we
33:16
also multiply it with FI, which are the fraction of tokens which is routed to each expert. And this is the quantity
33:22
which we try to minimize. This makes sure that the uh PI
33:29
is uniform. So expert importance will be similar. But it also makes sure that tokens are routed uniformly across
33:37
experts. Now the introduction of this load balancing loss presents some other challenges which DeepS seek tried to
33:43
solve and they made one very major change in the load balancing loss which we are going to see in one of the
33:49
subsequent lectures. But to understand that lecture, it's very important for all of you to understand load balancing
33:55
loss and auxiliary loss. The last concept which we are going to look at today is something which is called as
34:01
the capacity factor. So let's start understanding about this now. So one more guard rail which we can
34:09
implement to make sure that uh u to some expert does not hog all the limelight or
34:16
some expert does not get a huge number of tokens like in this case which we saw this expert was getting all the tokens
34:22
right to avoid this we can also implement something which is called as the capacity
34:27
factor. So let's say we have not in included capacity factor and something
34:33
like this happens right the next day is and if these tokens are all routed to one expert that's not a good sign for us
34:41
because it means that expert number two and expert number three are not getting any of the tokens. So we introduce
34:47
something called as the expert capacity or the capacity factor and that's given by this formula. So expert capacity is
34:54
tokens per batch divided by number of experts into the capacity factor. So
34:59
expert capacity just means how many maximum number of tokens one particular expert can handle. If capacity factor is
35:07
equal to one then it just means the number of tokens every expert can handle is the tokens per batch divided with the
35:13
number of experts. So if I have thousand if I have thousand tokens per batch if I
35:20
have thousand tokens per batch and if my number of experts is equal to four it means that every expert can handle only
35:27
250 tokens. So expert capacity is the maximum number of tokens which every expert can handle and increasing the
35:34
capacity factor make sure that every expert has more room to handle more number of tokens. And how do we
35:42
calculate the tokens per batch? It of course depends on the context or the batch size. It depends on the sequence
35:48
length which is number of tokens which are passed into one context and it depends on the top K. Why is there your
35:55
top K? Because remember that every token will be routed to two experts or K
36:00
experts. Right? So uh the number of tokens are actually
36:05
duplicated based on this top K factor which is there. If let's say I have top
36:11
K equal to two. If top k equal to two and I have E1, E2, E3 and E4, it means
36:17
that my one token is now routed to two of these, right? My one token is routed
36:22
to two of these. So every token will be routed to two. So that's why we have to multiply the tokens per batch with the
36:28
top K factor, right? So the expert capacity is tokens per batch divided by
36:33
number of experts into capacity factor. So if capacity factor is equal to one
36:39
then every expert gets an E1 split of tokens. If capacity factor is greater
36:46
than one then every expert can handle more than its fair share. So which means that some experts can have more tokens
36:52
while other experts it's fine if there are less number of tokens. And that may happen in load balancing right? If you
36:59
do load balancing it's not guaranteed that every expert will have same number of tokens. Some experts may have higher
37:06
end to endurance. So a capacity factor if a capacity factor is huge it might lead to
37:11
a lot of imbalance right which means one expert might hog the limelight. So generally what I've seen is capacity
37:17
factor is in the range of 1.125 to two etc. The capacity factor never goes more
37:23
than two otherwise one expert can hog all the limelight and uh we try to solve
37:28
this problem in today's lecture right that no one expert should have all the power. So it does not make sense to
37:34
include exceed the capacity factor by a huge amount. Whereas if capacity factor is less than one it means that we'll
37:41
have to drop off some tokens because if every if you limit what every expert can
37:46
handle to less than the total number of tokens then we'll have to drop off some tokens and that's also done sometimes
37:52
when language models are trained. It's not uh preferred to drop off tokens. So
37:59
for example when deepseek v3 came out the mixture of experts model which they have they did not drop off any single
38:06
token. So in this case when expert capacity is not implemented one expert
38:12
can have four tokens like this right but if expert capacity is implemented if that is equal to two if expert capacity
38:19
is equal to two it means maximum number of tokens one expert can handle is equal to two right. So this actually means
38:26
that expert number one cannot have four tokens. So if we implement expert
38:32
capacity, expert number one will never be able to have four tokens. So we'll never get some imbalance situation such
38:38
as this. Expert capacity is actually implemented in many modern language
38:43
models. I don't think deepseek version 3 implemented this but they have a novel
38:49
architecture which we are going to come to later. But first it was very important for all of us to understand
38:55
step one to step nine of how mixture of experts actually work. So in the previous lecture I showed you until step
39:02
number seven. I showed you until step number seven. And in today's lecture I covered three things. I covered the auxiliary
39:09
loss. Then I covered something which is called as load balancing. And then after
39:14
that I covered u after that I covered capacity factor. So there are three things which all of
39:20
us have learned today. Auxiliary loss, load balancing and capacity factor. All of these mechanisms are in place to make
39:26
sure that experts get a balanced load of tokens. So neither one expert should hog
39:32
the limelight and we should not have experts which are not doing anything. So we want to make sure that
39:38
experts have relatively equal importance. Right? And that's where auxiliary loss comes into picture. But
39:44
at the same time you also want to make sure that the fraction of tokens routed to experts is uniform. Uh and that's
39:52
where uh load balancing comes into the picture and expert capacity just limits the
39:58
maximum number of tokens one particular expert can handle through this capacity
40:05
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














