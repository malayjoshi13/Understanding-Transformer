# Understanding-Transformer

This repository is a dedicated effort to facilitate a deep dive for me (and hopefully you) into the PyTorch implementation (right from scratch!!!) of the complete architecture, training pipeline and inference pipeline of the vanilla Transformer from [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper.

This repo also facilitates understanding of the Transformer's practical use-case by letting the reader/user use it for the ***English to Hindi Language Translation*** task like below:<br>
English input: `Indian batting is at its peak now and even the bigger targets seem small to them.` <br>
Hindi translation: `भारतीय बल्लेबाजी इस समय चरम पर है और बड़े से बडा लक्ष्य भी उसके सामने बौना साबित हो रहा है।` (this output is not produced by model trained in this repo)
<p align="center">
  <img src="https://github.com/malayjoshi13/Understanding-Transformer/assets/71775151/b632b4ba-825b-4549-8b06-ac6e8b9f5497" width="430" height="500">
</p>

Without any further delay let's deep dive into the code implementation of vanilla Transformer right from scratch!!
    
## Repository Structure (file names in order of their usage for building architecture from scratch, training and inference)
- **config.py** --> file containing configurations like paths of folders and hyperparameters that need to be set (or use default ones) before training and inference

- **model.py** --> file having architectural components of the transformer in form of separate classes, like Word Embedding layer, Positional Encoding layer, Layer Normalization layer, FeedForward layer, Residual Connection, Multi-head attention layer, Encoder layer, Decoder layer, Linear layer (or pre-softmax layer). Towards the end, this file has a function which combines all individual components to build and initialize a Transformer network.

- **dataset_processor.py** --> file having components which pre-process the raw input data to make it usable for training and inference.

- **training_pipeline.py** --> file having pipeline to get processed data from [dataset_processor.py]() file and then train and validate the transformer model on it.

- **train_controller.ipynb** --> all-in-one file for training transformer (built from scratch!!) which clones this repository to get the required pipeline files for training, installs required packages and allows training (via [training_pipeline.py](https://github.com/malayjoshi13/Understanding-Transformer/blob/main/training_pipeline.py) file) using Colab's free computational resources. 

- **inference_pipeline.py** --> file having pipeline to do inference using the weights of trained transformer.

- **inference_controller.ipynb** --> all-in-one file for inferencing/transaltion using trained transformer via [inference_pipeline.py](https://github.com/malayjoshi13/Understanding-Transformer/blob/main/inference_pipeline.py) file.
  
- **visualize_attention.py** --> helps in visualising attention plots to understand all things it has learned during training.

## Configuration
Before starting training and inference, configure (or keep by default configurations) the following parameters in ```config.py``` file:
- batch_size,
- num_epochs = number of epochs,
- lr = learning rate,
- seq_len = max allowed sequence length,
- d_model = embedding size of model,
- datasource = name of translation dataset from HuggingFace on which you want to train your transformer,
- train_data_size = portion you want to choose out of the original dataset for training + validation,
- lang_src = source language for translation,
- lang_tgt = target language for translation,
- model_folder = folder path where you want to save model's weight,
- model_basename = name of model weight,
- preload = set "latest" to resume training from latest epoch, set epoch like 01 to resume training from say 01th epoch, or set None to do training from scratch,
- tokenizer_file = path of the folder where you want to save tokenizer,
- experiment_name = path of the folder where you want to save loggings for tensorboard. You can plot these loggings to visualize training and validation plots.

## Train
**Recommend Way:** [training_controller.ipynb](https://github.com/malayjoshi13/Understanding-Transformer/blob/main/training_controller.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/malayjoshi13/Understanding-Transformer/blob/main/training_controller.ipynb)

***Note:*** While going with this GColab-based training file, you can only set configurations using `config.py` file once you have cloned the GitHub repo within this GColab file's space.

**Alternate Way:** 
```
!git clone https://github.com/malayjoshi13/Understanding-Transformer.git
cd Understanding-Transformer
conda create --name transformer
conda activate transformer
pip install -r requirements.txt
!python training_pipeline.py
```
In both ways, in the end, you will get:
- `output/vocab` folder having tokenization files for language pairs you train your model on (English and Hindi for this case).
- `output/weights` folder having model training checkpoints.
- `runs` folder having tensorboard logging files. You can plot these loggings to visualize training and validation plots.

Dataset is prepared by taking data points from index 4000 to 3,4000 of [iitb-english-hindi](https://huggingface.co/datasets/cfilt/iitb-english-hindi) dataset's train split. This dataset has English-Hindi language pairs. Training happens on `90%` of this dataset and validation on `rest 10%`. Why used this dataset? As this dataset makes it easier to debug and play around as I speak English and Hindi languages. 

This was the configuration used during training and validation:<br>
batch_size: 24, <br>
num_epochs: 100, <br>
lr: 10**-4, <br>
seq_len: 350, <br>
d_model: 512, <br>
train_data_size_start: 4000, <br>
train_data_size_end: 34000, <br>
lang_src: "en", <br>
lang_tgt: "hi" <br>

## Training Results

Original results of Vanilla Transformer trained from the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762):

| Language-pair | BLEU score | Dataset |
| --- | --- | --- |
| English to German translation task | **28.4** | WMT-14 val |
| English to French translation task | **41.8** | WMT-14 val |
 
What I did (for now) is after coding architecture and training pipeline for vanilla Transformer, I trained the model using [training_controller.ipynb](https://github.com/malayjoshi13/Understanding-Transformer/blob/main/training_controller.ipynb) file which at the back use [training_pipeline.py](https://github.com/malayjoshi13/Understanding-Transformer/blob/main/training_pipeline.py) file. Due to computation constraints (as using Colab's free-tier GPU) training happened in phases. In first phase, model trained till 8th epoch and 63% of 9th Epoch. In second phase model started re-training from 9th epoch.

How I did training in two phases?<br>
I really don't advise this approach and would suggest going for the Colab Pro version if you can afford it. If you are just a student like me with limited financial resources, just follow this. So, after the first phase, when training stopped after the completion of the 8th Epoch, I switched to my other Google account, opened the training_controller.ipynb file there, and once GitHub was cloned in my GDrive, I moved the `output` and `tensorboard_logging` folders in the cloned repo at the switched GDrive.

`To avoid overfitting training was stopped after 15th Epoch. Training weight at Epoch 10 is used for inference (will use it in next section).`<br>

Here are the training and validation results in my case:

| Language-pair | BLEU score | CER | WER | Training loss | Validation loss | Dataset |
| --- | --- | --- | --- | --- | --- | --- |
| English to Hindi translation task | **0.61** (Epoch 10), **0.59** (Epoch 11) | **0.16** (Epoch 10), **0.19** (Epoch 11) | **0.35** (Epoch 10), **0.37** (Epoch 11) | **1.507** (Epoch 10), **1.46** (Epoch 11) | **1.533** (Epoch 10), **1.51** (Epoch 11) | [iitb-english-hindi](https://huggingface.co/datasets/cfilt/iitb-english-hindi)'s test split (from index 4000 to 3,4000) |

; here BLEU score: Ranges between 0 and 1. The closer the value is to 1, the better the translation. <br>
CER: Ranges between 0 and 1. The closer the value is to 0, the better the translation. <br>
WER: Ranges between 0 and 1. The closer the value is to 0, the better the translation. <br>

Training and Validation plots:

- Avg Batch Train Loss vs Epoch and Avg Batch Validation loss vs Epoch:<br>
  ![TrainLoss_ValidationLoss_Epoch](https://github.com/malayjoshi13/Understanding-Transformer/assets/71775151/36a5fa4d-61ea-447a-8fd6-1810ce730522)<br>
  
- Validation BLEU score vs Epoch:<br>
![valBLEU_epoch](https://github.com/malayjoshi13/Understanding-Transformer/assets/71775151/1f1f74ff-fa17-4ddd-8b2c-6966db0de56e)<br>

- Validation CER vs Epoch:<br>
![valCER_epoch](https://github.com/malayjoshi13/Understanding-Transformer/assets/71775151/8cad35b0-d2e7-40dc-bfe4-89081e3d5ffa)<br>

- Validation WER vs Epoch:<br>
![valWER_epoch](https://github.com/malayjoshi13/Understanding-Transformer/assets/71775151/f655f4cb-f8e6-4dd4-8924-3fc7b6435ba2)<br>

***Note:*** In this whole process, I didn't chase the training results shared in the [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper because the aim of this whole exploration is to better understand the main crux of architecture and the inner workings of vanilla Transformer (from a code implementation perspective). Soon, I'll also train it on WMT-14 and try to dive further into advanced training and scaling techniques. But till then I request not to compare my training results with original results.

## Inference (Translating from English to Hindi languag)
Use following notebook to translate: [inference_controller.ipynb](https://github.com/malayjoshi13/Understanding-Transformer/blob/main/inference_controller.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/malayjoshi13/Understanding-Transformer/blob/main/inference_controller.ipynb)

This script will automatically download the model's best weight (as of now it is weight at epoch 10) and tokenizers, will do translation on your given English input sentence and will allow you to visualize how each attention head learns like below:

<p align="center">
  <img src="https://github.com/malayjoshi13/Understanding-Transformer/assets/71775151/a531f06f-2266-49cf-b057-687a09681b92" width="600" height="1000">
</p>

## Acknowledgement
This whole exploration and implementation is nspired by following codes and blogs:
- [Attention Is All You Need paper](https://arxiv.org/abs/1706.03762)
- [The Original Transformer (PyTorch) by Aleksa Gordic](https://github.com/gordicaleksa/pytorch-original-transformer)
- [The Annotated Transformer by Harvard NLP](https://nlp.seas.harvard.edu/2018/04/03/attention.html)

Also a big thanks to the maintainers of the [iitb-english-hindi](https://huggingface.co/datasets/cfilt/iitb-english-hindi) dataset!!

## End-note
Thank you for patiently reading till here. I am pretty sure just like me, you would have also learnt something new about the Transformer's implementation. Using these learnt concepts I will push myself to solve failure cases similar to the one stated below and scale this repo further to other language pairs and NLP use cases. I encourage you also to do the same!!

In practice, I would suggest using the official PyTorch implementation or HuggingFace's Transformer models which have many more tricks for better scaling. This repo is not the best option for a deploy-friendly use case as my goal was to understand better the theory about vanilla Attention by implementing the architecture described in the [Attention Is All You Need paper](https://arxiv.org/abs/1706.03762) from scratch by taking some reference of existing codes and blogs wherever needed. Therefore, I have kept my exploration and implementations till the model was trained correctly and produced correct translation results. I did not aim to reproduce the results from the paper, nor to implement all of the bells and whistles. 

## Contributing
You are welcome to contribute to the repository with your PRs. In case of query or feedback, please write to me at 13.malayjoshi@gmail.com or https://www.linkedin.com/in/malayjoshi13/.

## Licence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/malayjoshi13/Understanding-Transformer/blob/main/LICENSE)
