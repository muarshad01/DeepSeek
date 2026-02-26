#### Integer and Binary Positional Encodings (PE)

1. Integer Positional Encoding (IPE)
2. Binary Positional Encoding (BRE)
3. Sinusoidal Positional Encoding (SinPE)
3. Rotary Positional Encoding (RoPE)

***

* 5:00

#### Why do we need positional embedding or positional encoding?
* The main reason is that the position at which a word comes in a sentence is very important to the context of the sentence.
* __Input embedding__ = Token embedding (Capture Semantic Information) + Positional embedding

```
The dog chased another dog.
```

* After coming out of the attention block, we want the model to understand the context of the sentence. We want the model to understand that the first dog who is chasing is different from the second dog who is being chased. Right?

* [Huggingface example](https://huggingface.co/blog/designing-positional-encoding)


#### Integer Positional Encoding (IPE)

```
The dog chased the ball. It couldn't catch it.
```

* 20:00

#### Binary Positional Encoding (BPE)

* Lower indices oscillate fast between positions. Higher indices oscillate slow between posoitons.

***
