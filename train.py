# this is my first transformer from scratch. I wanted to explore this topic with the 
# rise in popularity of chat-gpt, which is based on a transformer model. It basically
# predicts the next words after a question, and was trained on the internet. This is very
# time consuming and costly to do, so i will make a transformer model that will produce
# shakespeare text, with the data for training from the tensorflow website (tiny shakespeare)

# libraries needed
import torch
from model import Transformer_model

# params
device = 'mps' # mps/cpu/cuda

# functions used

# creating a function to get the bacthes of data
def get_batch(split):

    batch_size = 64
    
    # deciding if we have got train or test data
    data = train if split == 'train' else test

    # getting a random integer to start from in the data, batch times
    index = torch.randint(len(data) - context_len, (batch_size,))

    # getting the tensor of data by splitting our data using the
    # random indexs and context length
    x = torch.stack([data[i:i+context_len] for i in index])
    y = torch.stack([data[i+1: i+1+context_len] for i in index])

    # putting our training data on our gpu
    x, y = x.to(device), y.to(device)

    # returning our tensors
    return x, y


# first lets read in our dataset
with open("data.txt", 'r') as f:
    data = f.read()

# lets get all the different characters that can be generated in our model by
# looking at all the characters in our dataset.
vocab = sorted(list(set(data))) 
vocab_len = len(vocab)

# we can now tokenize our characters to be used in our model
# below we use character tokenizing, which is a very simple tokenizer
# next time i might use sub word tokens, which is a trade of from using full words
# and just characters 
# (libraries to help with this can be: tiktoken (OpenAI) and sentencePiece (Google))
# for this task of generating shakespeare, this tokenization is enough

# we first get a dictionary for the tokens going backwards and forwards
tokens = {char:i for i, char in enumerate(vocab)}
chars = {i:char for i, char in enumerate(vocab)}

# we now create these simple lambda functions that encode and decode our text
# by getting the value of the above dictionaries
encode = lambda text: [tokens[i] for i in text]
decode = lambda ints: ''.join( [chars[i] for i in ints] )

# lets split our data now so we have some testing to do at the end

# we can convert our data to a tensor so it is formatted correctly
data = torch.tensor(encode(data), dtype = torch.long) # long datatype is int64

# we can get the percent of data by multiplying the len by a decimal
percent = int((len(data)*0.9)) # 90 percent of the data, int used to get whole numbers for slicing

# lets use 90 percent for training and only 10 percent for testing later
train = data[:percent]
test = data[percent:]


# now before we train our model lets set some params initially
context_len = 256 # this is how many characters we input into the transformer to get an output

# initialising our model
model = Transformer_model(vocab_len, context_len)

# moving the model on our gpu
model = model.to(device)

# lets now train our model
model.train_model(get_batch)

# lets save the model
torch.save(model, "model1.model")

# setting the start of the script as a new line (array of 1,1 with zero)
start_context = torch.zeros((1,1), dtype = torch.long).to(device)

# printing the generation of 100 characters
print( decode( model.generate( start_context, tok_gen_len = 1000, context_len = context_len)[0].tolist() ) )

with open("output.txt", 'w') as f:
    f.write(decode( model.generate( start_context, tok_gen_len = 10000, context_len = context_len)[0].tolist() ))
