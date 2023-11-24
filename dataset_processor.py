import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    # This class defines and processes the dataset to be used for training the transformer model.

    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()

        # Define the sequence length
        self.seq_len = seq_len

        # Define the dataset
        self.dataset = dataset

        # Define the tokenizers for encoder and deocder side
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        
        # Define the source and target languages
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # tokenizer_tgt.token_to_id("[SOS]") , converts "SOS" into its corresponding id. 
        # This id is then converted in a tensor using "torch.tensor()" fn.
        # Repeat same process as above to get token ids for "[EOS]" and "[PAD]" tokens also. 
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64) # signed 64-bit integer is from -2^63 to 2^63 - 1, here one bit is used to represent the sign (positive or negative) of the number.
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        # First we pick each source-target sentence pairs.
        src_target_pair = self.dataset[idx]

        # Then we seperate the source and target sentences.
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Then we transform the sentences (going into encoder and decoder parts) into tokens 
        # and assign each token a unique id (its first step before embedding and positional encoding).
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Padding tokens are needed when our seq-length is lets say set to 10, but our current sentence is only 7 tokens long. 
        # Thus to make every sentence of same length, i.e. we add "padding tokens". Here we will add 3 padding tokens.
        # Calculating how many padding tokens we need by subtracting:- the digit "2" (as we have eos and sos tokens in each data sentence input to encoder) 
        # and the number of tokens in each sentence, from the "sequence length" we have set in the config file
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  
        # Here subtracting by "1" as we only add eos token while passing input in case of decoder.
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure the number of needed padding tokens is not negative. If it is, then it means sentence is too long.
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Adding encoding for sos token, sentence's tokens, eos token, and padding tokens to the input to fed to encoder. 
        # One thing to note: "[self.pad_token] * enc_num_padding_tokens" means if enc_num_padding_tokens = 3, then we will need 3 padding tokens. 
        # Thus, [self.pad_token] * 3 = [self.pad_token, self.pad_token, self.pad_token] 
        # Like: <enc_of_sos> + enc_of_Hi + enc_of_how + enc_of_are + enc_of_you + enc_of_? + <enc_of_eos> + <enc_of_pad_token> + <enc_of_pad_token> + <enc_of_pad_token>
        encoder_input = torch.cat( 
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Preparing input to be fed to decoder. Here we don't add "eos" token.
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Here add "eos" token to input. This thing will used as "label" to match with output of decoder.
        # How? See "decoder_input" at first iteration will be <eos> and at this time "label" will be <eos> + sentence's first token.
        # "decoder_input" will be fed to decoder and decoder will be expected to generate some token.
        # This something generated token will be matched and compared with "label" to see how close generated token is to "sentence's first token" (aka true token)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (Dim: seq_len)
            "decoder_input": decoder_input,  # (Dim: seq_len)

            # Defining mask for encoder. This mask prevents padding tokens to be seen by the encoder block.
            #
            # Lets see how its made:
            #
            # "encoder_input != self.pad_token": This part of the code is comparing each element of the encoder_input tensor with the value self.pad_token. 
            # It will return a boolean tensor, where each element is True if the corresponding element in encoder_input is not equal to self.pad_token, and False otherwise.
            # 
            # Using unsqueeze(0) adds a new dimension at the beginning of the tensor. 
            # For example, if you have a 1D tensor A=[1, 4, 3] having Dim: (3), then using unsqueeze(0) two times convert it to A=[[[1, 4, 3]]] having Dim: (1,1,3).
            # This "unsqueeze(0)" operation is often used when you need to align tensors for operations like element-wise multiplication or broadcasting.
            # For example, to do element-wise multiplication between A = torch.tensor([[1, 2, 3], [4, 5, 6]]) and B = torch.tensor([[[7, 8, 9]]]), then you need to first do A = A.unsqueeze(0) and then result = A * B.
            #
            # ".int()": It converts the boolean tensor (from the previous step) to an integer tensor, where True is represented as 1, and False is represented as 0.
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (Dim:1, 1, seq_len)

            # Defining mask for decoder.
            # 
            # In this we use same tensor like above as first part of the deocder mask. This first part, prevents padding tokens to be seen by the decoder block.
            # For second part we use causal_mask. See below to see what's causal mask. 
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            
            "label": label,  # (Dim: seq_len)

            "src_text": src_text,
            "tgt_text": tgt_text,
        }
    
def causal_mask(size):
    # Causal mask basically prevent decoder to see tokens of future.
    # Lets see how is it made.
    #
    # Ex: torch.ones(3, 3) will create tensor with all values as "1": |1, 1, 1|
    #                                                                 |1, 1, 1|
    #                                                                 |1, 1, 1|
    # 
    # then torch.triu(tensor, diagonal=1) will convert above tensor to a Upper Triangular tensor (where upper half triangle is non-zero): |0, 1, 1|
    #                                                                                                                                     |0, 0, 1|
    #                                                                                                                                     |0, 0, 0|
    #
    # if it was torch.triu(tensor, diagonal=0), then: |1, 1, 1| , because it means make lower triangle zero excluding the main diagonal.
    #                                                 |0, 1, 1|
    #                                                 |0, 0, 1|
    #
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)

    # but for masking we need a mask which is opposite to what we have generated above (tensor with upper part non-zero).
    # mask == 0 performs an element-wise comparison, it checks which elements in the mask tensor are equal to 0. 
    # As a result, it returns a "Boolean tensor" of the same shape where each element is "True" if the corresponding element in the mask is 0 and "False" otherwise. 
    # 
    # In this way we reverse the mask generated above and got a mask where lower half is "1" and upper part is "0".
    return mask == 0