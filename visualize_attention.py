import torch
import torch.nn as nn
from config import get_config
from training_pipeline import get_model, get_ds
import altair as alt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from tokenizers import Tokenizer
from model import build_transformer

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Get data and intialise transformer model
config = get_config()
# tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
# tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))
# model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config["seq_len"], config['seq_len'], d_model=config['d_model']).to(device)

train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt, _ = get_ds(config)
# model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

# Load the latest pretrained weights
model_folder = f"{config['model_folder']}"
model_filename = f"{config['model_basename']}*"
weights_files = list(Path(model_folder).glob(model_filename))
weights_files.sort()
model_filename = str(weights_files[-1])

state = torch.load(model_filename)
# model.load_state_dict(state['model_state_dict'])
seq_len = config['seq_len']

####################################

# This fn gets attention scores from encoder's attention-head, decoder's attention-head and encoder-decoder attention-head.
def get_attn_map(model, attn_type: str, layer: int, head: int):
    if attn_type == "encoder":
        attn = model.encoder.layers[layer].self_attention_block.attention_scores
    elif attn_type == "decoder":
        attn = model.decoder.layers[layer].self_attention_block.attention_scores
    elif attn_type == "encoder-decoder":
        attn = model.decoder.layers[layer].cross_attention_block.attention_scores
    return attn[0, head].data

####################################

# All three fns below help in creating the attention map.

def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
    return pd.DataFrame(
        [
            (
                r,
                c,
                float(m[r, c]),
                "%.3d %s" % (r, row_tokens[r] if len(row_tokens) > r else "<blank>"),
                "%.3d %s" % (c, col_tokens[c] if len(col_tokens) > c else "<blank>"),
            )
            for r in range(m.shape[0])
            for c in range(m.shape[1])
            if r < max_row and c < max_col
        ],
        columns=["row", "column", "value", "row_token", "col_token"],
    )

def create_attn_map(model, attn_type, layer, head, row_tokens, col_tokens, max_sentence_len):
    df = mtx2df(
        get_attn_map(model, attn_type, layer, head),
        max_sentence_len,
        max_sentence_len,
        row_tokens,
        col_tokens,
    )
    return (
        alt.Chart(data=df)
        .mark_rect()
        .encode(
            x=alt.X("col_token", axis=alt.Axis(title="")),
            y=alt.Y("row_token", axis=alt.Axis(title="")),
            color="value",
            tooltip=["row", "column", "value", "row_token", "col_token"],
        )
        #.title(f"Layer {layer} Head {head}")
        .properties(height=400, width=400, title=f"Layer {layer} Head {head}")
        .interactive()
    )

####################################

def load_next_batch(input_sentence, output_sentence):
    # #Load a sample batch from the validation set
    # batch = next(iter(val_dataloader))
    # encoder_input = batch["encoder_input"].to(device)
    # decoder_input = batch["decoder_input"].to(device)

    ####################################
    #replacment of above three lines because now we are not getting data from val_dataset but directly 
    source = tokenizer_src.encode(input_sentence)
    encoder_input = torch.cat( 
        [
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64), 
            torch.tensor(source.ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(source.ids) - 2), dtype=torch.int64)
        ], dim=0,
    )

    encoder_input=encoder_input.unsqueeze(0).to(device)

    target = tokenizer_tgt.encode(output_sentence)
    decoder_input = torch.cat(
        [
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64), 
            torch.tensor(target.ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(target.ids) - 1), dtype=torch.int64)
        ], dim=0,
    )
    
    decoder_input=decoder_input.unsqueeze(0).to(device)

    ####################################

    encoder_input_tokens = [tokenizer_src.id_to_token(idx) for idx in encoder_input[0].cpu().numpy()]
    decoder_input_tokens = [tokenizer_tgt.id_to_token(idx) for idx in decoder_input[0].cpu().numpy()]

    # check that the batch size is 1
    assert encoder_input.size(
        0) == 1, "Batch size must be 1 for validation"
    
    return encoder_input_tokens, decoder_input_tokens

####################################

def generate_attention_maps(model, input_sentence:str, output_sentence:str, attn_type: str):
    encoder_input_tokens, decoder_input_tokens = load_next_batch(input_sentence, output_sentence)

    max_sentence_len = encoder_input_tokens.index("[PAD]")

    # Show attention results for all 6-layers at encoder-side as well as 6-layers at decoder-side
    layers = [0, 1, 2]
    # And for all 8 single-heads of each encoder-block at encoder-side as well as 8 singleheads of each decoder-block at decoder-side
    heads = [0, 1, 2, 3, 4, 5, 6, 7]    

    if attn_type=="encoder":
        row_tokens = encoder_input_tokens
        col_tokens = encoder_input_tokens
    elif attn_type=="decoder":
        row_tokens = decoder_input_tokens
        col_tokens = decoder_input_tokens
    elif attn_type=="encoder-decoder":
        row_tokens = encoder_input_tokens
        col_tokens = decoder_input_tokens
    
    ####################################

    charts = []
    for layer in layers:
        rowCharts = []
        for head in heads:
            rowCharts.append(create_attn_map(model, attn_type, layer, head, row_tokens, col_tokens, max_sentence_len))
        charts.append(alt.hconcat(*rowCharts))
    return alt.vconcat(*charts)
