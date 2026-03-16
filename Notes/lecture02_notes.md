#### What is DeepSeek
* DeepSeek is a Chinese company, which builds Large Language Models (LLMs)

***

* 5:00

#### Emergent Properties
* An ability is emergent if it is not present in smaller models but is present in larger models.

***

* 15:00

| Model | Parameters |
|---|---|
| DeepSeek V3 (Foundational Model)| 671 B |
| DeepSeek R1 (Reasoning Model) | |
| LLama (Meta) | 70B |
#### Comparision with other AI models

* DeepSeek versus GPT-4
* DeepSeek versus [LLama](https://www.llama.com/)

***

#### Innovative Architecture
1. MLA
2. MoE
3. MTP
4. RoPE
5. Quantization

#### Training Methodology
* RL to teach complex reasoning to the model
* Rule based reward system
* Group Relative Policy Optimization (GRPO)

#### GPU Training
* NVIDIA Paralle Thread Execution (PTX)

* [DeepSeek's AI breakthrough bypasses industry-standard CUDA for some functions, uses Nvidia's assembly-like PTX programming instead News](https://www.tomshardware.com/tech-industry/artificial-intelligence/deepseeks-ai-breakthrough-bypasses-industry-standard-cuda-uses-assembly-like-ptx-programming-instead)

#### Model Ecosystem
* Model sistillation into smaller models (even down to 1.5B parameters)



equirements which they use to train the models so it's
32:43
like it it was the first proof that even small startups even companies which are
32:49
not maybe as funded as open a can build a large language model which is which is
32:55
awesome right which performs at parage with gp4 it and it does so with far less
33:04
resources what also happened with this is that people got scared right investors got
33:09
scared um there was a huge dip in the US tech US tech stocks in January 2025
33:15
because of deep six advancement the main reason was the idea of a lowcost open source Chinese AI
33:23
model threaten The Profit models of open AI Microsoft and even Google
33:28
and raised concerns about the AI supply chain and GPO markets one model or one company deep
33:36
seek brought about so many changes and that's what makes deep seek a turning point in history one major thing which I
33:43
believe has also happened because of deep seek is that developing countries such as for example India have started
33:49
have heavily started investing into building their own large scale foundational models if China's deep SE
33:55
can do it then why not other countries why only us and why only companies
34:01
coming out of us rather right why not other companies with resources which deep SE used can build their own
34:07
foundational models so all of this discussion has been started in fact the Indian government has also U released a call
34:15
for building foundational models that's pretty awesome I think and it's one of the main motivating factors for me to
34:21
create this series now let's come to the last section of today's lecture which is our
34:27
plan for this lecture series we have developed or I have rather divided this
34:32
lecture series into four phases based on what is the speciality about deep seek
34:38
the first phase for us is going to be um going into the architecture so
34:44
first I'll start with the attention mechanism then I'll go into multi-ad latent attention mixture of experts
34:50
multi token prediction quantization and rotary positional encodings I am going to assume that you have some amount of
34:56
knowledge of attention if not you can check the build llm from scratch Series so this series is going to be a bit more
35:03
advanced and it assumes that you you go through that previous series before I am going to start at a slightly higher
35:09
level here then we are going to go in the phase two in the training methodology phase three GPU optimization
35:16
tcks this will be a small phase I won't be having too many lectures here and then I'll conclude with lectures on
35:23
distillation a bulk of the lectures will be on phase number number one and phase number two and smaller number of
35:30
lectures on phase three and phase four so this is the main plan which we'll be following for the
35:36

***

series let me quickly summarize what we learned today first we looked at large
35:41
language models and the fact that they are engines of probabilistic next token
35:47
prediction we saw that size is a very important factor in large language models there is a size scaling law as
35:54
the size increases the models get better and better they start developing emerging emergent properties which are
36:00
not present in smaller models then we saw that the llm secret source is essentially Transformer the Transformer
36:08
architecture and then finally we saw that creating an llm means building a foundational model which is essentially
36:15
the pre-training stage and then we have a fine tuning stage there are two parts
36:21
then we saw that although deeps R1 has become popular the deeps company started
36:27
long back they started with the deeps llm first which is version one then they had version two and ultimately version three
36:34
which was a huge model 671 billion parameter and then ultimately they made deeps car1 which is a reasoning model
36:41
and that broke the internet why did it break the internet because deeps car1 U
36:47
has comparable performance to open AI stop model and at a fraction of the Cost Plus its open
36:54
source so deep seek is equally performed as GPT 4 their pricing is literally I
37:00
think 100 to 500 times less as I showed you in this lecture and finally it's fully open
37:07
source um strength and weaknesses the biggest strengths of deep seek are that
37:12
it's open source it's cost efficient and it has competitive performance so three big strength the biggest weakness might
37:19
be that it's not maybe as polished or safe as let's say gp4 or BL another weakness is that if you're
37:26
planning to deploy it locally or planning to use it securely you need to have infrastructure for downloading and
37:32
using a 671 billion 671 billion parameter model then we saw what makes
37:38
deep seek so special and there are four key ingredients here the first is the Innovative architecture training
37:45
methodology GPU optimization and model ecosystem within the training or within
37:51
the Innovative architecture we have five key things multi-head latent attention mixture of experts
37:58
multi- toen prediction quantization and rotary positional encodings then uh in the training
38:05
methodology we have the fact that they used large scale reinforcement learning to teach complex reasoning to the model
38:12
and they used a rule-based reward system uh which is also known as group relative
38:18
policy optimization rather than relying on human label data in the GPU optimization tricks they
38:25
used parallel thread exec PTA instead of Cuda only in some places I believe and
38:31
then finally they have a strong model ecosystem where they distill their main model into smaller models as low as 1.5
38:39
billion parameters in this lecture series we are going to follow the same workflow we'll
38:45
go with the first phase which is innovative architecture then we'll go to the second phase which is training
38:51
methodology then we'll go to the third phase which is GPU optimization tricks then we'll go to the fourth phase which
38:58
is model ecosystem I am going to assume a a good amount of knowledge about llms
39:04
and I will explain the attention mechanism again but I'll essentially start from the attention mechanism and
39:09
then dive into the details if you're a complete beginer I recommend the build llm from scratch series
39:17
first um and then finally I believe deep seek is a turning point in history because they literally showed that even
39:24
developing countries can build their own foundational model um if we are smart about the
39:31
Innovative architecture if we are creative um we can build a foundational model which is as good as let's say open
39:39
a models and that to at a low cost
39:44
and fully open source so they are truly democratizing AI that way so thanks a
39:50
lot everyone and uh I look forward to seeing you during the next lecture thank
39:55
you

***
