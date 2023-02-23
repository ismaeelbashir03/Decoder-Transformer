# Decoder-Transformer
This is transformer (without encoder) from scratch with torch

This code is based on the 'Attention is all you need' Paper: https://arxiv.org/abs/1706.03762

![diagram of transformer model](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png)

some adjustments were made to the original diagram above (pre layer norm, no encoder)

I implemented the Decoder part of the paper to generate text in a style of what it is trained in (shakespeare)
(Use a gpu as the model is quite big, to test on cpu, lower the number of head, blocks and embeddings etc.)
