import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
# This class corresponds to embedding layer which takes input data of dim (batch, seq_len) comprising of input sentences.

# During forward pass, input data "x" is passed through embedding layer resulting in a "embedding tensor" of shape (batch, vocab_size, d_model).
# here:
# - batch is the number of sequences being processed simultaneously
# - vocab_size is the total number of tokens in each sentence
# - d_model is the dimension of the embedding vector for each token

# This output "embedding tensor" has embeddings for each token of input sentences + embedding for padding token.
# It is then added with positional-encoding matrix and then resultant matrix gets used in Encoder and Decoder blocks.

# Note: Here role of padding tokens is that if we have set maximum length of each sentence to be 10, but at some instance number 
# of tokens in the sentence is less than 10 lets say 4 tokens, then we will add 6 "padding-tokens" to make length of sentence to be 10.
 
    def __init__(self, d_model: int, vocab_size: int) -> None:
        
        # super().__init__() is used in the constructor of "InputEmbeddings" class [which is a child class of "nn.Module"] to call 
        # the constructor of its parent class i.e. "nn.Module" class.
        # This ensures that while calling "InputEmbeddings" class in future, all the attributes from the parent class aka nn.Module are first initialized 
        # and then the attributes defined in the child class are properly initialized.
        super().__init__()

        self.d_model = d_model # d_model = Embedding vector size = 512 
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) # using torch's Embedding layer to generate embedding matrix of shape (vocab_size, d_model), i.e. (30000, 512)

    def forward(self, x):
        # As mentioned in the paper, generated embedding matrix is multiplied by sqrt(d_model) to scale the embeddings.
        return self.embedding(x) * math.sqrt(self.d_model)

##################

class PositionalEncoding(nn.Module):
# This class corresponds to positional encoding layer which during forward pass generates PE matrix and adds it to embedding matix.

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create an empty "positional encoding" matrix of shape (seq_len, d_model). 
        # Later on every empty space in this matrix will be filled with positional encoding value corresponding to that position.
        pe = torch.zeros(seq_len, d_model)

        # Create a vector of shape (seq_len, 1). 
        # Let us understand how it works:
        # Suppose seq_len is 5. The torch.arange(0, seq_len, dtype=torch.float) part will create a 1D tensor like this: [0.0, 1.0, 2.0, 3.0, 4.0].
        # Then, when you apply .unsqueeze(1) to it, the tensor is modified to have two dimensions. The result is a 2D tensor with one column and five rows (a column vector):
        # position = [[0.0],
        #[1.0],
        #[2.0],
        #[3.0],
        #[4.0]]
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # Create denominator for PE's formula.
        # Let us understand how it works:
        # "torch.arange(0, d_model, 2).float()" will create a 1D tensor like: [0.0, 2.0, 4.0, 6.0, 8.0].
        # This is similar to the "2i" part in PE's formula. Here i = 0, 1, 2, 3, 4 which are the indices of each embedding vectors
        # Next, 
        # "-math.log(10000.0) / d_model)" simulates "1/d_model" part. 
        # We have used additonal term of "-math.log(10000.0)" alongwith "1/d_model" part, to avoid numerical instability for large values of 1/d_model
        # Next, 
        # "torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)" simulates "2i / d_model" part
        # Next,
        # "torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))" aka "e^(2i * -log(10000) / d_model)" part.
        # This simulates "10000^(2i * 1/ d_model)" or "10000^(2i / d_model)" part.
        denominator = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices, i.e. pick all rows from each column of the pe tensor at even indices (0, 2, 4, ...) 
        # and replace the values in each row with sin-based PE values.
        # Here, pe[:, .......] is for pe's rows and pe[...., 0::2] is for pe's columns. 
        # Thus, pe[:, .......] select rows from each column of the pe tensor.
        # And pe[...., 0::2] select every consecutive columns (i.e. skipping one-one column as 2 is the "step size") starting from 0th index till end of column-dimension.
        pe[:, 0::2] = torch.sin(position * denominator) # sin(position * (10000 ** (2i / d_model))

        # Apply cosine to odd indices, i.e. pick all rows from each column of the pe tensor at odd indices (1, 3, 5, ...)
        # and replace the values in each row with cos-based PE values.
        pe[:, 1::2] = torch.cos(position * denominator) # cos(position * (10000 ** (2i / d_model))

        # Add a new dimension at the 1st position (i.e. index 0) of the PE matrix so that in future "number of batches" can come at that place.
        pe = pe.unsqueeze(0) # at the moment pe is of shape (1, seq_len, d_model), but in future pe will be of shape (batch, seq_len, d_model)

        # Register the positional encoding as a buffer so that it will be saved with the model and can be used everytime when needed
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Positional-encoding matrix "pe" getting added to input matrix "x" (which is the embedding matrix for each sentence)

        # Now as we already know sentence passed to encoder/decoder can have some padding-tokens in addition to actual tokens (to make all sentences of same length),
        # thus we need to transform shape of current "pe" matrix, i.e dim of tranformed "pe" matrix: 
        # [same "batch-count" as in current "pe" matrix, max sentence length which is upto the 2nd dim of x matrix, same "d_model" as in current "pe" matrix]

        # "requires_grad_" is set to False as this ensures during backpropagation we don't compute gradients 
        # for the positional encoding matrix as its a fixed thing which we don't want model to update/learn
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)

##################

class LayerNormalization(nn.Module):
# This class corresponds to Layer Normalization which during forward pass will normalize the input data "x". 

# Normalisation don't affect dim (batch, seq_len, hidden_size) of "x". 
    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps # value of epsilon is initialised above and then used in rest part of this "LayerNormalization" class to prevent dividing by zero or when std dev is very small
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter. torch.ones(5) creates a 1D tensor with 5 ones: [1, 1, 1, 1, 1].
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter. torch.zeros(5) creates a 1D tensor with 5 zeroes: [0, 0, 0, 0, 0].
        # Intution behind using alpha and bias is that in starting we consider that normalization is needed thus in starting we keep alpha and bias to be removed 
        # from formula : normalized_value = (alpha * (x - mean)) / (std + eps) + bias.
        # This we do by initialising alpha and bias to be 1 and 0 respectively.
        # Thus, normalized_value = (1 * (x - mean)) / (std + eps) + 0 = (x - mean) / (std + eps) = getting normalized value.

        # Then we leave this upon training process that if it will find normalization useful, then alpha and bias remains to be (or close to) 1 and 0 respectively.
        # If training process don't find normalization to be useful, then alpha and bias will have some value which will cancel out affect of normalization.

    def forward(self, x):
        # Calculating mean
        # argument of "dim=-1" means that the mean is computed along the last dimension of the "x", i.e. each token's feature column of dimenion = hidden_size.
        # argument of "keepdim=True" means the resulting tensor aka "mean" will have the same number of dimensions as the input tensor aka "x", 
        # but the dimension along which the mean will be calculated will be "1" (as it's a "single value" calculated for a each column corresponding to features of every token).
        # As a result, dim of "x" is (batch, seq_len, hidden_size) and dim of "mean" is (batch, seq_len, 1).
        mean = x.mean(dim = -1, keepdim = True)   
        
        # Calculating std. deviation
        # dim of std. deviation value is (batch, seq_len, 1) as it's a "single value" calculated for a whole column (corresponding to features of a single token)
        std = x.std(dim = -1, keepdim = True) 
        
        # here we have just implemented the theoritical formula of normalization using mean and std deviation.
        normalized_value = (self.alpha * (x - mean)) / (std + self.eps) + self.bias
        return normalized_value

##################

class FeedForwardBlock(nn.Module):
# This class corresponds to Feed-Forward layer which during foward pass will convert input data "x" of dim (batch, seq_len, d_model) into dim (batch, seq_len, d_ff) and back into dim (batch, seq_len, d_model).

# It do this by using torch's 2 linear layers and one dropout layer.
# Intutiion behind doing this is that in this process of linear transformation, weights w1, b1, w2, b2 gets learned which help in better highlighting of each token's features.

# Input "x" of dim (batch, seq_len, d_model) gets passed to linear1 layer. 
# Output of linear1 layer (with w1 and b1 as learnable parameters) of dim (batch, seq_len, d_ff) gets passed to linear2 layer (with w2 and b2 as learnable parameters).
# Output of linear2 layer is the final output of dim (batch, seq_len, d_model).
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # here in feedforward block, first input data "x" goes to linear1 layer then to relu then to layer2 layer.
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

##################

class ResidualConnection(nn.Module):
# This layer corresponds to Residual connection which will add input data "x" to the output of sublayer.

# Note that this sub-layer takes input of "x" and outputs some processed form of "x".
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

##################

class MultiHeadAttentionBlock(nn.Module):
# This layer corresponds to Multi-head attention layer.

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        assert d_model % h == 0, "d_model is not divisible by h" # Make sure d_model is divisible by h otherwise each head will not get a part of the input

        self.d_k = d_model // h # Dimension of splitted Q, K, V matrices as seen by each head. Consider dk = dq = dv

        self.w_q = nn.Linear(d_model, d_model, bias=False) # defining Wq matrix
        self.w_k = nn.Linear(d_model, d_model, bias=False) # defining Wk matrix
        self.w_v = nn.Linear(d_model, d_model, bias=False) # defining Wv matrix

        self.w_o = nn.Linear(d_model, d_model, bias=False) # defining Wo matrix

        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1] # d_k is equal to last dimension of "query" matrix passed to this attention function which is actually equal to d_model//h.

        # Now, just apply the formula from the paper of calculating attention score: 

        # Firstly matrix-multiplication of q matrix with k-tranpose matrix and then divide it by square root of "d_k --> whichis 64 in our case"
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # here (batch, h, seq_len, d_k) --> (Dim of attention_scores: batch, h, seq_len, seq_len)

        # Second step is to apply mask
        if mask is not None:
            # Write value close to -inf to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)

        # Third step is to apply softmax. No change in dim, its still (batch, h, seq_len, seq_len).
        attention_scores = attention_scores.softmax(dim=-1) 

        # Fourth step is to apply dropout (optional)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Last step is to multiply attention scores (Dim: batch, h, seq_len, seq_len) with "value" matrix (Dim: batch, h, seq_len, d_k). Resulting in (Dim: batch, h, seq_len, d_k)
        final_output = attention_scores @ value

        # Return attention scores and final output of attention-layer's single-head. 
        return final_output, attention_scores

    def forward(self, q, k, v, mask):
        # multiplying input data's copy named as "q" (Dim: batch, seq_len, d_model) with weight matrix "wq" (Dim: batch, d_model, d_model) to get "query" matrix (Dim: batch, seq_len, d_model)
        query = self.w_q(q)
        # same here as above but now to get "key" matrix 
        key = self.w_k(k) 
        # same here as above but now to get "value" matrix
        value = self.w_v(v) 

        # Dividing "query", "key", "value" matrix into number of parts (in our case 8 parts) to send to each of the heads. 
        # Lets understand how this breaking happens through "query" matrix:
        # We use "view" to break tensors by doing overall reshaping of tensors.
        # In this breaking process, we don't want first and second dim of tensors to change, thus we simply write query.shape[0], query.shape[1].
        # We only want to change the last dim of the tensor as it will get break. 
        # Thus, overall data dim will change like (batch, seq_len, d_model) --> (batch, seq_len, h, d_k).
        # Then we use transpose to change the order of dimensions from  (batch, seq_len, h, d_k) to (batch, h, seq_len, d_k).
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2) # thus Dim of "query" matrix is (batch, h, seq_len, d_k), i.e. (batch, 8, 5, 64)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2) # Dim of "key" matrix is (batch, h, seq_len, d_k), i.e. (batch, 8, 5, 64)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2) # Dim of "value" matrix is (batch, h, seq_len, d_k), i.e. (batch, 8, 5, 64)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together.
        # For this we first use "transpose" function to do (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k).
        # Then using "view" function we do (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # At last multiply x with Wo matrix by doing (w_o) X (x). Resultant tensor is output of a single "multi-head attention layer". 
        # In this whole process, we started with Dim of x: batch, seq_len, d_model. 
        # This x gets copied three and muliplied with w_q, w_k, w_v weight matrices to get query, key and value matrices.
        # And after few more steps we get resultant tensor of Dim: batch, seq_len, d_model.
        resultant_tensor = self.w_o(x)  
        return resultant_tensor








# Using different parts defined above to create Encoder part, Decoder part and then combine them to create Transformer model.


class EncoderBlock(nn.Module):
    # This class defines a base architecture of every Encoder block in Encoder part.

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)]) # initialise and save two Residual layers in this list to be used below

    def forward(self, x, src_mask):
        # Input embedding matrix "x" gets copied three times and along with "src_mask" get feed into attention_block layer via the 
        # fn "lambda x: self.self_attention_block(x, x, x, src_mask)".
        # Then output of fn "lambda x: self...." and the input "x" goes to first residual layer via 
        # fn "residual_connections[0](x, lambda x: se....)". There both gets added.
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        
        # Then output of 1st residual layer that we got from fn "residual_connections[0](x, lambda x: self.se....." and output of feedforward layer goes to second residual layer.
        # There both gets added.
        # This output is the final output of each encoder block.
        final_encoder_output = self.residual_connections[1](x, self.feed_forward_block)
        return final_encoder_output
    
class Encoder(nn.Module):
    # This class defines architecture of whole Encoder side.

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        # Output of one encoder block will be onput for next encoder block, thus keep doing this for all encoder blocks 
        # and at last returning the output embedding matrix "x" of last encoder block.
        for layer in self.layers:
            x = layer(x, mask)
        return x

##################

class DecoderBlock(nn.Module):
    # This class defines a base architecture of every Decoder block in Decoder part.

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)]) # in each decoder block, we need three residual connection layers.

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # First part of decoder block comprise of self-attention layer and residual connection. 
        # As it is self-attention thus all input values are decoder's input only i.e. "x".
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # Second part of decoder block comprising of cross-attention layer and residual connection. 
        # As it is cross-attention thus one decoder input "x" is used for q matrix and encoder's output is used for k and v matrices.
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        # Third part of decoder block comprise of feedforward layer and residual connection.
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    # This class defines architecture of whole Decoder side.

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x

##################

class ProjectionLayer(nn.Module): # aka linear layer before softmax layer at end of overall decoder side

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)

##################
    
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed # embedding generator fn for encoder side
        self.tgt_embed = tgt_embed # embedding generator fn for decoder side
        self.src_pos = src_pos # positional encoding generator fn for encoder side
        self.tgt_pos = tgt_pos # positional encoding generator fn for decoder side
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src_embd = self.src_embed(src) # generates embedding matrix for input tokens
        src_emd_pos = self.src_pos(src_embd) # generates positional encoding matrix for input tokens and add it to embedding matrix
        encoder_output = self.encoder(src_emd_pos, src_mask) # pass the embedding+pos_enc matrix (Dim: batch, seq_len, d_model) alongwith mask to encoder part
        return encoder_output 
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # same process as in encoder above 
        tgt_emb = self.tgt_embed(tgt)
        tgt_emb_pos = self.tgt_pos(tgt_emb)
        decoder_output = self.decoder(tgt_emb_pos, encoder_output, src_mask, tgt_mask)
        return decoder_output
    
    def project(self, x): # aka linear layer at end of decoder side
        return self.projection_layer(x) # converts decoder's output to tensor of Dim: (batch, seq_len, vocab_size)

##################









# Lets initiate Transformer model.


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks by initialising sub-layers inside every encoder block
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks by initialising sub-layers inside every decoder block
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the whole encoder and decoder sides
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer