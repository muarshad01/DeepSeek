* There are two major issue with MoE models, which we are going to look at in Steps 8 & 9.

#### __Step 8__: Auxiliary loss version
* In MoE models, the routing mechanism selects a subset of experts for each input.
* If some experts are chosen too often while others are underutilized,  it leads to inefficient learning and potential performance bottleneck.
* An auxiliary loss term is added to main training loss to penelize imbalanced expert selection, pushing the routing functin toward a more uniform distribution.
* To calcuate the auxiliary loss, we first start with expert selector weight matrix, which consistes of the experts assigned to every token and the probabilities assigned to every expert.

***

* 10:00




* __Step 9__: 

***

at all of these importance values, you'll see that expert number three
10:18
carries the most importance and expert number two carries the least importance. Why is this the case? Because expert
10:25
number three has of course more tokens routed to it with overall higher probabilities and expert number two has
10:31
only two tokens routed to it with overall lesser probabilities. That's why expert number three is more
10:38
important than expert number two and also expert number one. Okay. So this is the concept of
10:44
expert importance and this intuitive terminology of expert importance is
10:50
exactly what we are going to utilize to calculate this auxiliary loss term. Right? So ideally what do I want? I want
10:59
a balanced model. Right? I want a balanced I want a balanced mixture of
11:04
experts model. And what does having a balanced mixture of experts model mean? It means
11:11
that all of the experts which I have all of the experts such as E1, E2, E3
11:17
roughly all of these experts should have equal importance. It should not be the case
11:23
that one of these experts has a huge amount of importance and another expert is completely neglected. If an expert is
11:30
completely neglected, it means that no token will be routed to that expert and we don't want such a situation. What
11:37
will happen if no token is routed to an expert? That token will that expert will sit completely idle and that expert will
11:44
not learn anything. This leads to inefficient training and it also leads to inefficient inference. So ideally
11:52
what I want is that these expert importance values I want these values to
11:57
be roughly similar for all of my experts. So what we are going to do is
12:03
that we are going to penalize the mixture of experts model if the expert importance scores have a lot of
12:10
variation this because this would mean that some experts are more important than others. So essentially what we are
12:17
going to do is that we are going to construct a loss. We are going to construct a loss term. That loss term
12:23
will be very high if so this loss term will be high if E1, E2 and E3 have a lot
12:32
of variation which means they are very different from each other and that loss term will be low if E1, E2 and E3 are
12:39
roughly similar to each other. So what I'm going to do is that I'm going to track a quantity which is called as the
12:46
coefficient of variation. Coefficient of variation is a statistical concept and
12:51
it's given by the simple formula. You just take the standard deviation and you divide by the mean. That's g that gives
12:57
you this quantity which is called as CV or coefficient of variation. So what I'm going to do now is that I have three
13:04
values right? Expert importance of E1 is 1.4. Expert importance of E2 is 1 and
13:10
expert importance of E3 is equal to 1.6. I'm going to use these three values and
13:16
I'm going to find the coefficient of importance of these three. How will you do this? You will first find the
13:21
standard deviation of these three values. You'll find the mean of these and you will divide the standard
13:26
deviation by the mean. If this value is higher, it means the coefficient of variation is very high and the values
13:33
are very different from each other. If this value is low, it means the coefficient of variation is low and the
13:39
values are roughly similar to each other. And what you actually want is you want
13:44
these values to be similar to each other so that the coefficient of variation is smaller. So we are going to define the
13:52
auxiliary loss term and that loss term is the scaling factor which is lambda into the coefficient of variation whole
13:59
squared. So my auxiliary loss my auxiliary loss I'm going to
14:05
define as lambda uh which is my scaling factor. It's a hyperparameter which we
14:10
need to fix multiplied with the coefficient of variation whole squared. So higher the coefficient of
14:17
variation higher is going to be my auxiliary loss and lower the coefficient of variation lower is going to be my
14:23
auxiliary loss. And so this auxiliary loss term is the term which I'm going to
14:28
add to the LLM training loss. So the LLM training loss is
14:35
originally defined for the next token prediction, right? And this can be a categorical cross entropy loss,
14:41
perplexity loss etc. We are going to add the auxiliary loss term to this. And the
14:46
amount of importance which needs to be given to the auxiliary loss is quantified by this hyperparameter which
14:52
is lambda. And this CV here is of course coefficient of variation. So in my
14:58
particular case the coefficient of variation turns out to be.187.
15:04
uh so we add the below terms to our LLM loss and that is the scaling factor multiplied by with the coefficient of
15:11
variation whole square. So what will this do? Addition of this loss term will actually ensure
15:18
that the expert importance is uniformly distributed among different experts. If
15:24
the expert importance is uniform across experts, the coefficient of coefficient of variation will naturally be small and
15:31
that would mean that experts have relatively u similar importance
15:37
values. Um so this is the first concept which I wanted to teach you today and that's the concept of auxiliary loss.
15:44
Auxiliary loss was implemented in mixture of experts so that all the experts which I have have relatively
15:51
similar importance. And what is importance? Importance is essentially we
15:56
look at the probability of the tokens uh probability of ex one expert
16:02
receiving the tokens and then we add all those together. Okay. Uh that's essentially
16:09
what quantifies the uh expert importance. And I want these expert importance to be roughly similar to each
16:15
other. So that I have a balanced uh mixture of experts model. That's the
16:21
first concept which is the concept of auxiliary loss. The second concept which I want to teach all of you today is
16:27
something which is called as load balancing. So it turns out that just having or just minimizing the auxiliary
16:33
loss is not enough to make sure that the u load is balanced across different
16:40
experts. we need to do one more thing and that is called as load balancing. So
16:45
while the previously discussed token expert importance is useful assigning
16:51
equal importance to experts does not necessarily lead to uniform token routing and this is a key concept.
16:58
Assigning equal Assigning equal importance to
17:07
tokens does not necessarily mean or does not necessarily mean uniform token
17:13
routing. And let me show this to you with a very simple example. Let's say we have those same
17:20
four experts or this example which I'm showing you right now is not necessarily a concrete example but it helps
17:26
illustrate the point. So when you take a look at the example, you'll see that the rows here do not sum
17:32
up to one, which they ideally should. But I'm just using this example to clarify this point that expert
17:39
importance is very very different than uh expert importance is very different
17:44
than uniform token routing. So let's say if you take a look at these two experts, right? If you take a look at expert
17:51
number one and expert number two and what I'm going to do is that first let me calculate the expert importance. What
17:57
do we do to calculate expert importance? We look at the columns and we add the column entries together. Right? So if
18:04
you look at the first column right now and you add all the entries, this sums up to one. So the expert importance for
18:10
expert number one is one. And if you look at the second column now and if you add all the entries of the second
18:16
column, these entries also sum up to one. Uh so the expert importance values of
18:22
expert one and expert number two are both equal to one. But take a look at
18:28
the tokens which are routed to these experts. Right? Expert number one has only one token which is routed to it.
18:34
This token number one. And for token number two, token number three and token number four are not at all routed to
18:40
expert number one. Whereas expert number two has four tokens routed to it. Token one, token two, token three and token
18:47
number four. This clearly illustrates the point that expert number one only has one token but
18:55
expert two is essentially hogging all the limelight. All the tokens are going
19:00
to expert number two but the expert importance is same for all for both these tokens.
19:06
So that clearly shows that just looking at the expert importance and making the expert importance same for different
19:13
experts will not make sure that the tokens are routed uniformly to different experts. We need to have some other uh
19:21
metric and that metric is calculated through something which is called as the load balancing
19:27
loss. So as I've mentioned over here if you add up the probability scores for both the experts both would sum up to
19:35
one. Hence both the experts have equal importance. Right? However, expert one
19:40
has fewer tokens routed with higher confidence. Whereas expert number two
19:45
has many more tokens routed to it but with far lower confidence. Right? Because the probability of tokens being
19:51
routed here is [Music] 0.25. Uh so what would be the issue if
19:57
the tokens which are routed to experts are non-uniform. So if the tokens which are sent to each
20:03
expert is non-uniform it can lead to a high memory usage and a reduced mixture of experts
20:10
performance. Um and as a result what we want to do is that we want to make sure the load of every expert is balanced.
20:18
And what do we mean the load of every expert? It means the number of tokens which are received by every expert that
20:23
needs to be balanced across the different experts. And that's why we implement something which is called as
20:29
the load balancing. To do load balancing there are essentially two things which we need to
20:34
calculate. The first thing which we will calculate is the probability that the router will choose a given expert and
20:40
this is called as pi. So the first thing which we'll have to calculate is the probability that the
20:47
router will select a given expert. And to do this calculation it's pretty
20:53
simple because we will just we are just going to use the expert importance scores. So for these three experts,
20:59
expert one, expert one, expert number two, and expert number three, the expert importance values are 1.4, 1.0, and 1.6.
21:08
Correct? So the probability that the router will select every expert
21:13
um or the probability that the router will select an expert for a given token is just going to be for expert number
21:20
one, it's 1.4 divided by 4. For expert number two, it's 1 divided by 4. And for
21:26
expert number three, it's 1.6 divided by 4. The reason we're dividing by four is because we want to have a probability,
21:33
right? So the probability that my router is going to choose expert number one is 1.4 divided by 4,
21:39
that's.35. The probability that the router is going to choose expert number two is 1 divided by 4. That's 25. And
21:48
the probability that the router will choose expert number three is actually
21:53
1.6 6 / 4 which is 04. So if you add these three it will sum up to
21:59
1. So it is important for all of you to intuitively know what pi means. So P1 P1
22:07
actually denotes the probability that the router will choose expert one. P2
22:12
denotes the probability that the router will choose expert 2 and P3 denotes the probability that the router is going to
22:18
choose expert number three. uh and these three probabilities add up to
22:24
one. Along with these probabilities, we are also going to find one more quantity
22:29
and that quantity is essentially the fraction of tokens which are dispatched to each
22:34
expert. And this is the most important quantity which needs to be calculated
22:39
along with the probability. Now let me illustrate why we calculate this quantity with a simple example. Let's
22:45
say we take the same case, right? We have expert number one and expert number two. um and let's say we have four
22:52
tokens token one token two token three and token number four so let's say this is my scenario number
22:58
one and I will also consider a scenario number two expert one expert two token one
23:06
token two token three and token four so let's say in scenario number one expert
23:11
one hogs all the limelight so I'm going to say token one passes goes to expert
23:17
one with one and expert two with zero token two goes to expert one with probability one and expert two with
23:22
zero. This is 1 0 and this is 1 0. And in the second case, token number one
23:28
goes to expert one with probability 0.5. This is so or let's say let me say this is
23:34
uh yeah let's say this is
23:41
64 this is uh 4 6 this is 6.4 4 and this is also 04
23:49
and 6. Now what I'm going to do is I'm going
23:55
to show you the calculation for the probability first and then the fraction of tokens which are routed to every
24:01
expert. Right? So in this case what is the so for E1 and E2 we are
24:08
going to find the probabilities and the fractions right. So this is P pi and
24:13
this is FI. This is BI and this is FI. So what's the probability that uh every
24:21
token tokens are routed to expert number one. So we first find the expert importance right that is four and that
24:28
is zero in this case and in this case the expert importance is 1 + 1 that is 2
24:34
and in this case it is two. So the probability that it will be routed to E1
24:41
is just 4 / 4 which is equal to 1. And the probability that it will be routed
24:46
to E2 is 0 / 4. In this case, probability that the tokens will be
24:51
routed to E1 is 2 / 4. And the probability that the tokens will be
24:57
routed to E2 is 2 / 4. Correct? And now FI is the fraction
25:03
of tokens which are routed to every expert. So let's say I have four tokens, right? How many of them are routed to
25:09
E1? All four of them. So the fraction of tokens is 4x4 and here it's 0 out of four. And in this case, what's the
25:16
fraction of tokens which are routed to E1? So token number one will be routed
25:22
to E1 because E1 has a higher probability than E2 and token number
25:28
three also will be routed to E1 because E1 has a higher probability than E2. So
25:33
two tokens out of the four are routed to E1. So the fraction of tokens routed to E1 are 2x4. And the fraction of tokens
25:41
which are routed to E2 are also 2x4. Okay. Now what we are going to do
25:48
is that the load loss which we are going to do it will be depending on a quantity
25:53
which is called as F_subi into PI. which would mean that we are going to look at f_sub_1 into p1 and f_sub_2
26:00
into p2 and we are going to minimize this quantity. I'll tell you why we minimize this quantity. But first let's
26:08
calculate this quantity for both of these. Right? Uh so this quantity is now
26:14
f_sub_1 into p1 plus f_sub_2 into p2. So for the first case what's f_sub_1 into
26:21
p1? That's 1 into 1 plus 0 into 0. So this loss is going to be 1. And for the
26:27
second case, what's f_sub_1 into p1? That's essentially 1x2 into 1x2 which is
26:33
1x4 plus 1x2 into 1x2 which is 1x4. So this loss is equal to half. So do you
26:39
see something special over here? Essentially what you see over here is that the loss for case number one is
26:46
higher than the loss for case number two. Which is what we wanted, right? Because case number one seems to be
26:52
highly unbalanced to expert one seems to be hogging all the limelight and expert
26:58
two is not getting any token at all. Whereas whereas case number two is much more balanced. Balanced in the sense
27:05
that the expert importance is same across the experts. That's fine. But also the fraction of the tokens routed
27:11
to the experts is also similar. Right? Expert number one is getting 1 by two. Expert number two is also getting 1x2.
27:19
So I'm getting a lower loss. So these are the two quantities which I consider in calculating this
27:26
loss which is also called as load balancing. And I will try to minimize this sigma f_subi into pi. Which means I
27:33
will get fi into pi for all my experts. I'll add them together and I'll try to reduce this quantity. What will happen
27:40
if I try to reduce f_subi into pi? It will mean that first of all it will mean
27:46
that I have a much more uniform distribution because as you see the more uniform the distribution you get this
27:53
product F1 P1 decreases whereas if one experts hogs the limelight this product
27:58
will go on increasing which I don't want. So when I minimize this quantity, I will make sure that FI is uniform and
28:06
PI is uniform across experts. And I want FI to be uniform and
28:11
I want PI also to be uniform. If PI is uniform that makes sure that the experts
28:16
have equal importance. But now I'm also saying that FI will be uniform and FI is
28:22
the fraction of tokens routed to that expert. So that satisfy that satisfies
28:28
the main thing which we started in this section right. Uh we saw that expert importance or auxiliary loss does not
28:35
necessarily mean that the token routing is also uniform. So to make sure that the token routing is uniform I include
28:42
one more factor which is FI and through this product minimization I make sure that FI is reduced and PI
28:49
also is reduced. Um so now I will show this a bit more formally. In our particular
28:56
case which we were seeing um we had we have already calculated PI right for
29:02
expert number one it's for P1 it's P1 is.35 P2 is 0.25 and P3 is 04. Now let's
29:10
calculate the fraction of the tokens dispatched. So wherever it's 0.5 we will choose either of the experts right. So
29:17
in the case of first token E2 is the expert which is selected. In the case of second token it's E1. In the case of
29:25
third token it's E3. And in the case of fourth token again it's going to be E1. So fact fraction of tokens dispatched to
29:31
expert one is 2 out of four which is half. Fraction of tokens dispatched to expert 2 is 1 out of four. And fraction
29:38
of tokens dispatched to expert 3 is again 1 out of four. So as we can see the f_sub_1 is.5 which is 2x4. F_sub_2
29:47
is 1x4 which is 255 and F_sub_3 is also 1x4 which is 255 and the load balancing
29:54
loss is as I mentioned we have to do sigma f_subi into pi and we have some scaling factor terms over here we have a
30:01
scaling factor and we also multiply with the number of experts which we have in
30:06
this case the number of experts is equal to three and the scaling factor is again a
30:11
hyperparameter which we have to choose uh so the load balancing loss is the scale scaling factor multiplied by the
30:18
number of experts multiplied with sigma f_si into pi where fi is the fraction of tokens sent to expert I and pi is the
30:26
fraction of the probability which is allocated to expert I which also can be thought of as expert
30:32
importance. So if you sum up fi into pi for these three tokens uh for these
30:38
three experts I'm going to assume a lambda which is a scaling factor of 0.01 01 3 which is the number of experts and
30:45
this is the sigma f_subi into pi. So this is f_sub_1 into p1 plus f_sub_2 into p2 plus f_sub_3 into p3 and that
30:52
gives me a total load balancing loss of 0.0101. This is the loss which I try to
30:58
minimize. Now you might be thinking based off this formula that how would minimizing this loss actually help us.
31:05
So mathematically this loss is minimized when the product f_subi into pi is minimized across all the experts. Right?
31:13
So the product will be minimal when both f_subi and pi are simultaneously low.
31:19
And what would mean when fi and pi are simultaneously low? If fi and pi are
31:24
simultaneously low, it just means that I'm getting to a more uniform distribution. Whereas if one of fi and
31:31
pi if f_si is very high, pi is very low, the to the distribution is not very uniform and so this loss will be very
31:38
high. So minimizing this loss is actually pushing me
31:44
uh so minimizing this loss is actually pushing me to reduce the overall imbalance uh between the assigned
31:50
importance and the actual routing. So what this minimization of the loss
31:56
term actually does is that it makes sure that uh my importance expert importance
32:02
and my routing are aligned with each other. Which means that whenever the expert importance is high, I'm also
32:08
routing my tokens there. Whenever the export expert importance is low, I'm als I'm routing less number of tokens over
32:15
there. So I just want my FI and PI to be aligned with each other and also simultaneously lower together. When fi
32:22
and pi is lower together, it means that my distribution is becoming more uniform. And when fi and pi are more
32:29
aligned to each other, it means that experts with higher importance handle more tokens. Experts with lower
32:36
importance handle less tokens. And so there is no mismatch. Overall this increases
32:41
uh more uh it increases the stability during training and uh this also leads
32:49
to a more efficient and balanced use of the mixture of experts architecture. Okay. So I hope you have understood the
32:56
difference between these two terminologies. Auxiliary loss is something which is completely different than the load balancing. In auxiliary
33:04
loss, we just look at the expert importance and we make sure the expert importance are equal among the different
33:10
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





