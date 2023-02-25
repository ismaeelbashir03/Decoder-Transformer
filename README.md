# Decoder-Transformer
This is transformer (without encoder) from scratch with torch

trained model weights after 5000 steps - loss: 1.5: https://drive.google.com/file/d/1Bkcl2UpZFk9bVzXBgnWqOZkEZaPpJfhn/view?usp=share_link

word prediction model 1 weights after 5000 steps - loss: 1.09: https://drive.google.com/file/d/1zafG_7eLMp2tIputQxqAe3qFXC2xBep-/view?usp=share_link

word prediction model 2 weights after 10000 steps - loss: 0.57: https://drive.google.com/file/d/1FTxeQf0NMWVSOLHVNWUNU5s6gdrYaqi0/view?usp=share_link

This code is based on the 'Attention is all you need' Paper: https://arxiv.org/abs/1706.03762

![diagram of transformer model](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png)

some adjustments were made to the original diagram above (pre layer norm, no encoder)

I implemented the Decoder part of the paper to generate text in a style of what it is trained in (shakespeare)
(Use a gpu as the model is quite big, to test on cpu, lower the number of head, blocks and embeddings etc.)

I also did a word tokenizer instead of a character one used for shakspeare and used the batman movie scripts. this
can be found in the word_model folder. - this is a more advanced version of ther script generator.

This decider is the pretraining for a model like chat-gpt. It can be fine tuned to become a question answering bot or become a sentiment analysis
or some other things. but this code is only for pretraining the model to complete text it was trained on.
