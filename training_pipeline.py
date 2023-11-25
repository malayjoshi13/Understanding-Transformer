import warnings
from tqdm import tqdm
import os
from pathlib import Path
import torchmetrics
from torch.utils.tensorboard import SummaryWriter

# from this repo
from model import build_transformer
from dataset_processor import BilingualDataset, causal_mask
from config import get_config

# from torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

# from Huggingface 
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

########################

def get_or_build_tokenizer(config, ds, lang):
    # This fn provides builds tokenizer.
    
    # Todo so it provides two option:
    # - first option: check if tokenizer is already saved in local at 'output/vocab/tokenizer_{specific_language}.json', if yes then load it from local.
    # - second option: if not, then calls tokenizer from huggingface, builds it and then save it in local system to be used later for tokenization.
    Path('./output/vocab').mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path(config['tokenizer_file'].format(lang)) # at {0}, lang_src gets add, like: output/vocab/tokenizer_en.json
    print(tokenizer_path)

    # first option
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    # second option
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

########################

def get_ds(config):
    # This fn gets the raw dataset and convert it into trainable format.

    # First load dataset using HuggingFace's "load_dataset" fn.
    # https://huggingface.co/datasets/cfilt/iitb-english-hindi need this: ds_raw = load_dataset(f"{config['datasource']}", split='train').
    # https://huggingface.co/datasets/opus_books/viewer/en-it need this: ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train').
    split = "train["+str(config['train_data_size_start'])+":"+str(config['train_data_size_end'])+"]"
    ds_raw = load_dataset(f"{config['datasource']}", split=split)
    print(len(ds_raw))

    # Then build tokenizers.
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Then keep 90% for training and 10% for validation.
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # Then ue "BilingualDataset" fn from dataset.py file to process the data.
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence.
    max_len_src = 0
    max_len_tgt = 0
    #
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    #
    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    
    # Define dataloader
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt, ds_raw

########################

def get_model(config, vocab_src_len, vocab_tgt_len):
    # This fn initialise the transformer model using "build_transformer" fn from model.py file. 

    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model 

########################

def train_model(config):
    # This fn defines process to train the transformer model.

    # Define the device to use out of cuda or cpu
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)
    print("Using device:", device)

    # Creating a sub-folder inside dataset folder to store model's weights.
    Path(f"{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    # Using "get_ds" fn defined above to get the train and validation data. 
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt, ds_raw = get_ds(config)

    # Passing the data to transformer model via "get_model" fn defined above. 
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # And defining the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # Logging the ongoing training results in Tensorboard under "experiment_name" panel.
    writer = SummaryWriter(config['experiment_name'])

    initial_epoch = 0
    global_iterations = 0

    # If the user specified a model to preload before training, load it else train from scratch (i.e every parameter initialized from starting).
    preload_value = config['preload']
    #
    #
    if preload_value == 'latest':
        print("Resuming training from latest epoch")
        model_folder = f"{config['model_folder']}"
        model_filename = f"{config['model_basename']}*"
        weights_files = list(Path(model_folder).glob(model_filename))
        if len(weights_files) == 0:
            model_filename = None
        
        else:
            weights_files.sort()
            model_filename = str(weights_files[-1])
    #
    elif preload_value: # here preload_value==epoch coz if you want to resume training from a particular epoch then in config do:- cfg['preload'] = 00, or cfg['preload'] = 01, etc.  
        print("Resuming training from a specific epoch")      
        model_folder = f"{config['model_folder']}"
        model_filename = f"{config['model_basename']}{preload_value}.pt"
        model_filename = str(Path('.') / model_folder / model_filename)
    #
    else:
        model_filename = None
    #
    #
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_iterations = state['global_iterations']
    else:
        print('No model to preload, starting from scratch')

    # Defining loss function.
    # Here "ignore_index=tokenizer_src.token_to_id('[PAD]')" --> ignores the padding tokens.
    # And "label_smoothing=0.1" --> avoids overfitting by dividing confidence that a model has on one label (which is closest to actual label) to other low probability classes.
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # Defining the training loop.
    for epoch in range(initial_epoch, config['num_epochs']):
                
        # Just prints five pairs anywhere in between the dataset to just check if data has been correctly loaded and processed by script.
        print("Printing random five data points to just check if data has been correctly loaded and processed by script.")
        if epoch==0:
            for i in range(20,25):
                src_target_pair = ds_raw[i]
                source_text = src_target_pair['translation'][config['lang_src']]
                target_text = src_target_pair['translation'][config['lang_tgt']]
                print(f"Source Sentence {i + 1}: {source_text}")
                print(f"Target Sentence {i + 1}: {target_text}")
                print("")
            print("#########################################################")
            print("")

        torch.cuda.empty_cache()
        
        print("starting training process")
        model.train()

        train_batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")

        sum_batch_train_loss = 0

        for batch in train_batch_iterator: # each batch has 25 training data points

            # Batching inputs and masks of the encoder and decoder.
            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)
            # Batch the label aka actual data
            label = batch['label'].to(device) # (B, seq_len)

            # Pass the input data through encoder, decoder and the projection layer and get the predicted output.
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compute the loss between predicted output (i.e. proj_output.view(-1, tokenizer_tgt.get_vocab_size())) and actual data (i.e. label.view(-1)).
            # Loss use a simple cross entropy. And because we are calculating loss per batch thus we call it "batch_train_loss".
            # Thus, in simple form, batch_train_loss = cross_entropy_loss(proj_output, label)
            batch_train_loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            train_batch_iterator.set_postfix({"train loss of current batch": f"{batch_train_loss.item():6.3f}"})

            # Summing up all batch train losses of a particular epoch to later find the average train loss of a particular epoch.
            sum_batch_train_loss += batch_train_loss.item()

            # Log the batch loss (i.e. loss of each batch of a particular epoch) in Tensorboard
            writer.add_scalar('train loss vs global iteration', batch_train_loss.item(), global_iterations)
            writer.flush()

            # Backpropagate the loss
            batch_train_loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # It tracks how many total iterations model has done. If each epoch has 1000 iterations, then at end of epoch 2 it will be 2000 global iterations.
            global_iterations += 1

        # Once all batches of a particular epoch gets processed and we are ready to move to next epoch, we do:
        #
        # 1. Compute average loss of a particular epoch
        epoch_train_loss = sum_batch_train_loss / len(train_dataloader)
        print({"avg train loss after each epoch": f"{epoch_train_loss:6.3f}"})
        writer.add_scalar('train loss vs global epoch', epoch_train_loss, epoch)
        writer.flush()
        #
        # 2. Run validation.
        print("#########################################################")
        print("starting validation process")
        do_validation(model, epoch_train_loss, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: train_batch_iterator.write(msg), epoch, global_iterations, writer)
        print("#########################################################")
        #
        # 3. Save the model weights and optimizer's state.
        # This helps to continue from exact epoch (we are not starting from exact batch of a particular epoch) where training earlier stopped on.
        model_folder = f"{config['model_folder']}"
        model_filename = f"{config['model_basename']}{epoch:02d}.pt"
        model_filename = str(Path('.') / model_folder / model_filename)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_iterations': global_iterations
        }, model_filename)

########################

def do_validation(model, epoch_train_loss, val_dataloader, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, epoch, global_iterations, writer, num_examples=5):
    # This function do validation after each training epoch to get us validation scores.

    # Telling the model that we are in validation mode.
    model.eval()

    # Will be used later to count the number of examples shown while validation goes on in background.
    example_counter = 0

    # Will be used later to store the source, target and predicted text.
    source_texts = []
    expected = []
    predicted = []
    expected_for_bleu = []
    final_expected_for_bleu = []

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # Get console window width.
    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    sum_val_loss = 0
    
    # Asking the model that don't calculate gradient now, we are now doing the validation.
    with torch.no_grad():

        # Pick each data point (as batch_size is set 1 for validation) in validation dataset right from starting till the end.
        # Then seperate source and target sentences from this picked data point. Generate predicted sentence. Append all these to lists.
        # Continue this process for all data points in validation dataset one-by-one (as batch_size=1).
        #
        # Parallel to this, every time validation process starts after an epoch ends, 
        # keep displaying first 5 examples (each example is made up of source, target and predicted sentence) on user's screen as part of validation process.
        #
        # Why are you getting 5 different examples everytime even when the all data points of same validation data is used in every validation process?
        # Thanks to "random" process with which sequence of data points is getting mixed 
        # and thus you are feeling you ar eseeing new validation data points; even when its not the case.
        for batch in val_dataloader: # each batch has 1 validation data point

            # Keep a track of how many data points have been shown on screen.
            example_counter += 1

            # Store encoder input and mask. Then check that the batch size shuld be 1 as its validation. 
            encoder_input = batch["encoder_input"].to(device) # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            #....................................................................................................................
            
            # This part is for validation loss

            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)
            label = batch['label'].to(device) # (B, seq_len)

            batch_val_loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

            # Summing up all val losses per validation data point to later find the average val loss across all validation data points at a particular epoch.
            sum_val_loss += batch_val_loss.item()

            # Not logging as logging val loss for every iteration makes sooo many loggings on tensorboard that whole Colab will get slow
            # Log the batch val loss (i.e. loss of each batch within a particular epoch) in Tensorboard
            # writer.add_scalar('val loss vs global iteration', batch_val_loss.item(), global_iterations)
            # writer.flush()

            #....................................................................................................................

            # This part is for BLEU, WER and CER

            # Getting the generated output aka "model_out" from the model via "greedy_decoder" fn 
            # This includes passing encoder input and mask to encoder, getting representations from encoder and passing it to decoder, 
            # getting the decoder output, projecting it to get the predicted output. 
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device)

            # Storing and printing the source, target and predicted texts.
            source_text = batch["src_text"][0]
            source_texts.append(source_text)
            #
            target_text = batch["tgt_text"][0]
            expected.append(target_text)
            expected_for_bleu.append(target_text)
            final_expected_for_bleu.append(expected_for_bleu)
            expected_for_bleu = []
            #
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            predicted.append(model_out_text)
            #
            # After printing "num_examples" trios of validation results, we stop printing further
            if example_counter == num_examples:
              print_msg('-'*console_width)
              print_msg(f"{f'SOURCE: ':>12}{source_text}")
              print_msg(f"{f'TARGET: ':>12}{target_text}")
              print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")
              print_msg('-'*console_width)
    
    # Once validation happen across all data validation points, then we do:
    #
    # 1. Find and log the average val loss per epoch in Tensorboard
    epoch_val_loss = sum_val_loss / len(val_dataloader)
    print({"avg val loss after each epoch": f"{epoch_val_loss:6.3f}"})
    writer.add_scalar('val loss vs global epoch', epoch_val_loss, epoch)
    writer.flush()
    #
    # 2. Plot "val_loss/epoch" and "train_loss/epoch" on same graph for comparison and checking overfitting or underfitting 
    # writer.add_scalar('loss', epoch_val_loss, epoch, color='purple')
    # writer.add_scalar('loss', epoch_train_loss, epoch, color='green')
    writer.add_scalars('loss', {'epoch_val_loss': epoch_val_loss, 'epoch_train_loss': epoch_train_loss}, epoch)
    writer.flush()
    #
    # print("predicted during val")
    # print(predicted)
    # print("ground truth during val")
    # print(expected)
    # print("ground truth during val for bleu")
    # print(final_expected_for_bleu)
    # #
    # 3. Compute the BLEU metric and log in Tensorboard
    metric = torchmetrics.text.BLEUScore()
    bleu = metric(predicted, final_expected_for_bleu)
    writer.add_scalar('validation BLEU vs global epoch', bleu, epoch)
    writer.flush()
    print("bleu "+str(bleu))
    #
    # 4. Compute the char error rate and log in Tensorboard
    metric = torchmetrics.text.CharErrorRate()
    cer = metric(predicted, expected)
    writer.add_scalar('validation cer vs global epoch', cer, epoch)
    writer.flush()
    print("cer "+str(cer))
    #
    # 5. Compute the word error rate and log in Tensorboard
    metric = torchmetrics.text.WordErrorRate()
    wer = metric(predicted, expected)
    writer.add_scalar('validation wer vs global epoch', wer, epoch) 
    writer.flush()
    print("cer "+str(wer))

########################

def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    # This function is used to generate the predicted output by the model by making encoder to output once 
    # and then again and again using that encoder output to predict next word via decoder.

    # Getting id for [SOS] and [EOS] tokens.
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Pre-compute the Encoder output once and then reuse it for every step in the decoding at Decoder
    encoder_output = model.encode(source, source_mask)

    # Start decoding process by initializing the decoder input with the sos token.
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

########################

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
