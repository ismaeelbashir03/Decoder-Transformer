the joker_output.txt file has a custom start sentence to start the generation. this can be implement yourself with the following code in python after loading the model:

```
# creating a custom scenario
start_context = torch.tensor([encode("speaker_heading: JOKER\n text: Batman walks in.".replace("\n", " NEWLINE ").replace(" ", " SPACE ").replace("(", "( ").split(" "))], device = device)

with open("joker_ouput.txt", 'w') as f:\n
    f.write(decode( model.generate(start_context, tok_gen_len = 1000, context_len = context_len)[0].tolist() ))
```
    
 
- inside the encode functions string you can edit the text, for a new line use '\n'. 
- And try to stick to the training format of starting with either (text:, speaker_heading:, dialog:, scene_heading:)
- leave the fomatting at the end with the replace functions and the split functions (this formats the text to be inputted into the model)
