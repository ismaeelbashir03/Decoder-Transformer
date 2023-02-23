# this is my first transformer from scratch. I wanted to explore this topic with the 
# rise in popularity of chat-gpt, which is based on a transformer model. It basically
# predicts the next words after a question, and was trained on the internet. This is very
# time consuming and costly to do, so i will make a transformer model that will produce
# shakespeare text, with the data for training from the tensorflow website (tiny shakespeare)

# libraries needed
import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'mps' # mps/cpu/cuda

# using a bigram language model, that uses a lookup table to make predictions
# not very good, loss ends up averaging to 2.4ish, no previous context used, only predicts
# next character based on the current only.
class Transformer_model(nn.Module):

    # we define our constructor, using pytorchs modules super constructor first
    def __init__(self, vocab_len, context_len) -> None:
        super().__init__()

        # we can define our number of embeddings for each vector
        num_embeddings = 384

        # here we create a table of vectors for each charcter, the vector size is
        # the num of embeddings (32). we can use this to get a embedding vector when 
        # inputting a character (value determines the probability for that charcter chosen next)
        self.token_embedding_table = nn.Embedding(vocab_len, num_embeddings) 

        # here we create another table of vectors, but this time for the position of the character
        # each position has a vector with len of number of embedding, values
        self.position_embedding_table = nn.Embedding(context_len, num_embeddings)

        # setting the number of heads for our multi head attention layer
        num_heads = 6

        # setting a param for the number of blocks in our transformer
        num_blocks = 6

        # here we initialise our blocks from the transformer model
        self.blocks = nn.Sequential(*[Block(num_embeddings, num_heads, context_len) for _ in range(num_blocks)])

        # here we add our final layer normalisation before the last linear layer
        self.final_ln = nn.LayerNorm(num_embeddings)

        # setting the language model head to a linear transformation changing our vector size
        # to the number of charcters in our vocab
        self.lm_head = nn.Linear(num_embeddings, vocab_len)

    # defining a function to feed data forward in the model, this function basically
    # gets a vector of vocab size for each charcter inputted, each element in the vector
    # corrosponds to a character next to be presented, if the value is higher, the more likely
    # it will be picked, (because we are using softmax to get the probability)
    def forward(self, index, y_true = None):

        # we can get the shape of the batch and time step
        B, T = index.shape

        # here we get our embedding vectors by looking up on the table our character 
        # index that we want to predict the next charcacter for
        token_embeddings = self.token_embedding_table(index) 

        # here we use our positional embedding vectors by using our lookup table
        # from 0 to the length of the time step, our output is going to be vectors of size
        # C, with T of them for each tiem step (T, C)
        pos_embeddings = self.position_embedding_table(torch.arange(T, device=device))

        # we can now add these embeddings together to get one vector embedding
        # this holds the embedding of the position of the character aswell as the character itself
        embeddings = token_embeddings + pos_embeddings

        # we can now feed this to the block layer
        embeddings = self.blocks(embeddings)

        # we finally pass our data to the layer normalisation in preparation for
        # the final layer
        embeddings = self.final_ln(embeddings)

        # we can now convert the Channels/vector size to the vocab len by using our lm head
        # this allows us to pick a charcter from the softmax distribution
        y_pred = self.lm_head(embeddings)

        # if we are evaluating there is no real y so no need for loss
        if y_true == None:
            loss = None
        else:
            
            # the logits we have are in the shape (Batch, Time, Channel), we need to change 
            # this for the loss function. time = context len, channel = vector len we can
            # just reduce the dimensions to fix this

            # getting the shape of each dimension
            B, T, C = y_pred.shape

            # joining the batch and time dimensions to make the tensor one dimension less
            # basically just gettin g abunch of vectors, and taking away the batch seperation
            y_pred = y_pred.view(B*T, C)

            # we now match the shape to our target charcter
            y_true = y_true.view(B*T) # no channel as real y has no vector, only truth

            # calculating our loss using cross entropy (neg log loss)
            loss = F.cross_entropy(y_pred, y_true)

        return y_pred, loss

    # creating a function to generate characters continuously 
    def generate(self, index, tok_gen_len, context_len): # index in shape (B, T)
        
        # looping for the token generation length
        for _ in range(tok_gen_len):

            # limiting the index to be within context size when we feed it into the model
            index_crop = index[:, -context_len:]

            # passing the current character index through the model to get the vector prediction
            pred, loss = self(index_crop)
            
            # we take the last time step's vector (B,T,C) (the char before final character predicted)
            pred = pred[:, -1, :]

            # using softmax activation to get the probability of the vector values
            probs = F.softmax(pred, dim = -1) # in (B, C) shape

            # getting the next character by using probability, output of shape (B, 1)
            next_char_index = torch.multinomial(probs, num_samples = 1) 

            # concatenating the next character to the index, increasing the time step
            # length
            index = torch.cat((index, next_char_index), dim = 1) # shape is (B, T+1)

        return index

    # function to get a mean loss instead of just single loss, this reduces noise
    # from random batches

    # we will not get the gradients for this so we can save space
    @torch.no_grad()
    def estimate_loss(self, iters, get_batch):

        # creating a dictionary for the output of train and test loss
        out = {}

        # runnning the eval mode for the model
        self.eval()

        # for each split of train and test
        for split in ['train', 'test']:

            # we initially set the losses to zero
            losses = torch.zeros(iters)

            # for each iteration
            for k in range(iters):
                
                # we get the batch data
                X, Y = get_batch(split)

                # we get the loss from feeding the data forward in the model
                pred, loss = self(X, Y)

                # we add each loss to the list of losses
                losses[k] = loss.item()

            # we get the mean of the losses
            out[split] = losses.mean()

        # we put the model back in train mode
        self.train()

        # we return the mean loss
        return out


    def train_model(self, get_batch):

        # optimiser used is AdamW, this is like the Adam optimiser, but it adds weight decay
        # so it incourages small weights and prevents overfitting and generalisation
        optimizer = torch.optim.AdamW(self.parameters(), lr = 3e-4)

        # initialising our batch size to be 32
        batch_size = 64

        # initialising our max iterations and every step we print of training loop
        max_iters = 5000
        iter_step = 2

        for iter in range(max_iters):

            # once in a while we will print mean loss of batches
            # instead of the loss at this point (could have lucky or unlucky batch)
            if iter % iter_step == 0:
                losses = self.estimate_loss(iter_step, get_batch)
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")


            # getting a batch of data
            x_batch, y_batch = get_batch('train')

            # making a prediction and getting loss
            pred, loss = self(x_batch, y_batch)

            # setting the gradients to initially None, for optimization (saves space)
            optimizer.zero_grad(set_to_none = True)

            # getting the gradients of the loss function of the model
            loss.backward()

            # optimizing the model with the gradients
            optimizer.step()

        # printing our loss
        print(f"loss: {loss.item()}")

# creating a class for teh attention head, this has query, key and value, values
# these are used when forwarding data in the model, we use them as follows:
# the key embedding is used to show what a token is in the context of itself and 
# what is represents. The query is an embedding of what the current token is looking 
# for (what its most important context is). The value embedding is what the curent token 
# wants to share about itself to the other tokens. We can put this all together to get a 
# embedding table that shows the attention of each token with the previous ones (we get only)
# the previous by using trill (explained more later). The attention value itself is calculated
# by doing the dot product of a current tokens query value by the other tokens key value, this 
# will give higher values for similiar embeddings and result in embeddings to have more of a 
# distribution when ran through softmax. we then can multiply the softmax distribution of the attention
# with the value embedding to give different data to different tokens, with the strength of the data
# determined by the strength of the weight of the key and query dot producted together.
# this class is one head of this attention system
class Head(nn.Module):

    def __init__(self, head_size, num_embeddings, context_len) -> None:
        super().__init__()

        # defining the key, query and value as linear transformations of size head_size
        # bias is not normally used for attention
        self.key = nn.Linear(num_embeddings, head_size, bias = False) 
        self.query = nn.Linear(num_embeddings, head_size, bias = False)
        self.value = nn.Linear(num_embeddings, head_size, bias = False)

        # we can now define how we get rid of attention from future tokens, because we 
        # are decoding we cannot see the future (if encoding we can remove this).
        # we do this by using tril, which returns an inputted matrix back with the 
        # top right triangle of data as zeros, this means that as we go down a row, 
        # for each vector, we only have non zero values for previous ones.
        self.register_buffer('tril', torch.tril(torch.ones(context_len, context_len))) 

        # we define a dropout value for the dropout layer
        dropout = 0.2

        # we can define a dropout layer
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):

        # we get the size of x
        B, T, C = x.shape

        # we forward through our key and query to get the embeddings 
        # for each time step token
        # output shape is (B,T,head_size)
        k = self.key(x)
        q = self.query(x)

        # we now calculate the attention for the tokens in the time step

        # we dot product the q and k, we transpose k (only for each batch), 
        # so we can actually do the dot product
        weights = q @ k.transpose(-2, -1)

        # we standardise the weights as when we use softmax, if we have such high 
        # variance of values the softmax distribution tends the higher value to 1 and 
        # the rest to just zero, so only 1 other token can have attention, if we scale down
        # the values we get more values that are not just tending 1 and zero adn they have
        # actual strength values which are useful. 
        # we do this by dividing by the head size square root
        weights = weights / (C**(0.5))

        # we now make sure future tokens have no meaning so they cant communicate ahead
        # we do this by getting the tril which is of size time step len by time step len,
        # and where this has values of zero we put negative infinity, so when we do softmax 
        # distribution, they get zero (no meaning, therefore cannot be used)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # we now get the distribution of the softmax of the weights to get values between 
        # 0 and 1 taht we can use to mulitply our value embedding by
        weights = F.softmax(weights, dim = -1) # we do this on the last dimension (-1)

        # we can apply our dropout to the weights
        weights = self.dropout(weights)

        # we now get the value embedding
        v = self.value(x)

        # finally we can dot product our value (what we want to share) by the weights
        # (how much we should share)
        output = weights @ v

        # returning our output embedding
        return output

# this gets the head class above and create a multi head attention layer for the model
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, num_embeddings, head_size, context_len) -> None:
        super().__init__()

        # defining the dropout of the final layer
        dropout = 0.2

        # we initialise our heads as the number of heads inputted with a size inputted
        self.heads = nn.ModuleList([Head(head_size, num_embeddings, context_len) for _ in range(num_heads)])

        # we now initialise the projection, which is just a linear transformation of the
        # previous layer
        self.projection = nn.Linear(num_embeddings, num_embeddings)

        # we can define a dropout layer
        self.dropout = nn.Dropout(dropout)

    # here we pass through our previous embedding to all our head layers
    def forward(self, x):
        
        # multi head attention layers just concatonate the heads output, so we
        # just do that to the heads
        x = torch.cat([h(x) for h in self.heads], dim = -1)

        # we can now forward through our projection
        x = self.dropout(self.projection(x)) # we dropout on the final layer

        # we return the output
        return x

# this is a simple mlp layer that we can use after the multihead attention layer in our model    
class FeedForward(nn.Module):

    def __init__(self, num_embedding) -> None:
        super().__init__()  

        # here we can define the dropout for our feed forward layer
        dropout = 0.2

        # here we use a simple nn which has one linear layer, with relu as the activation
        self.net = nn.Sequential(
            nn.Linear(num_embedding, 4 * num_embedding),
            nn.ReLU(),
            nn.Linear(4 * num_embedding, num_embedding), # this is our projection layer
            nn.Dropout(dropout)
        )

    # here we just pass the input to our mlp (NN)
    def forward(self, x):
        return self.net(x)

# here we define our block class, this is used to do a block in the transformer
# displayed in the 'attention is all you need paper', we made this a class to call 
# multiple instances of this block. This block can be simply defined as communication
# then computation
class Block(nn.Module):

    def __init__(self, num_embeddings, num_heads, context_len) -> None:
        super().__init__()

        # here we set the head size to the length of embeddings divided by the number of heads
        head_size = num_embeddings // num_heads

        # here we setup up multi head attention layer
        self.sa = MultiHeadAttention(num_heads, num_embeddings, head_size, context_len)

        # here we setup our feed forward layer
        self.ff = FeedForward(num_embeddings)

        # here we setup our layer normalisation layers
        # we will bw doing pre layer norms, which we means we do this before the
        # transformation (in the paper it is after the transformation)
        # (see layer norm class below for more info)
        self.ln1 = nn.LayerNorm(num_embeddings)
        self.ln2 = nn.LayerNorm(num_embeddings)
    
    # the forward module of this block is just passing through the input to our
    # self attention and nn
    def forward(self, x):

        # passing through our data through the block, we do some residual learning
        # this means that we add our input from before doing computation to the result of
        # the computation. We do this to make our model more optimisable since gradients
        # distribute equally over addition, we allow the gradient to have a quicker route 
        # to the previous part and add this to the gradient of doing all the computation 
        x = x + self.sa(self.ln1(x)) # we apply layer norm on the input
        x = x + self.sa(self.ln2(x)) # we apply layer norm on the input

        # returning our output
        return x

# here we create our layer normalisation, this a normalisation of inputs to a layer
# by getting mean and standard deviation and normalising the inputs before passing 
# them into the layer, this is similiar to batch normalisation, but is better in some 
# ways as the network can see different normalisations in one go and we can do training
# and testing normalisation the same because its normalised within the batch. Layer 
# normalisation is also better for RNN's, (transformer is also time step so benifit appplies)
#--- torch already has a layer norm function, i wrote this for learning ---#
''''class LayerNorm():

    def __init__(self, dim, eps = 1e-5):
        
        # initialising the epsilon value, gamma, and beta values
        self.eps = eps
        self.gamma = torch.zeros(dim)
        self.beta = torch.zeros(dim)

    # here we calculate the forward pass
    def __call__(self, x):
        
        # here we get the batch mean and variance, (by using the 1st dimension)
        x_mean = x.mean(1, keepdim = True)
        x_var = x.var(1, keepdim = True)

        # here we get the batch hat value, by subtracting the mean and dividing by
        # the variance (normalising), we add the epsilon value to avoid dividing by zero
        xhat = (x-x_mean) / torch.sqrt(x_var + self.eps)

        # we then multiply by the gamma and add the beta
        self.out = self.gamma * x.hat + self.beta

    def parameters(self):
        return [self.gamma, self.beta]'''