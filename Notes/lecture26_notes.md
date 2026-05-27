## DeepSeek Quantization
1. Mixed Precision Framework
2. Fine-grained quantization
3. Increasing accumulation precision
4. Mantissa over Exponents
5. Online Quantization

* FP8 Training (Section 3.3; page 14)
<p align="center">
  <img src="https://github.com/muarshad01/DeepSeek/blob/main/images/lec26/deepseek-quantization.png" width="600" height="300" />
</p>


| Research Paper |
|---|
| [DeepSeek-V3 - Jan 2025](https://arxiv.org/pdf/2412.19437)||
| [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale - Nov 2022](https://arxiv.org/abs/2208.07339) |
  * Outliers

***

#### Quantization
* [Blog: A Visual Guide to Quantization by Maarten Grootendorst](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)
  
* Quantization aims to reduce the precision of a model's parameter from higher bit-widths (like 32-bit floating point) to lower bit-widths (like 8-bit integers).

***


#### FP32 to INT8

|   | Symmetric Quantization |
|---|---|
| scale            | $\frac{\text{max}(\lvert x \rvert)}{127}$ |
| Quantiztion (q)  | $round \bigg( \frac{x}{scale} \bigg)$ | 
| Dequantizqation | $q \times scale$ |

***


***

* [TurboQuant: Redefining AI efficiency with extreme compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
