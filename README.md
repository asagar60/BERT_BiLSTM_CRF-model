# Keyphrase Extraction using BERT + BiLSTM + CRF (Semeval 2017)

Deep Keyphrase extraction using BERT + BiLSTM + CRF .

## Usage

1. Clone this repository and install `pytorch-pretrained-BERT`
2. From `scibert` repo, untar the weights (rename their weight dump file to `pytorch_model.bin`) and vocab file into a new folder `model`.
3. Change the parameters accordingly in `experiments/base_model/params.json`. We recommend keeping batch size of 4 and sequence length of 512, with 6 epochs, if GPU's VRAM is around 11 GB.
4. For training, run the command `python train.py --data_dir data/task1/ --bert_model_dir model/ --model_dir experiments/base_model`
5. For eval, run the command, `python evaluate.py --data_dir data/task1/ --bert_model_dir model/ --model_dir experiments/base_model --restore_file best`

## Results

### Subtask 1: Keyphrase Boundary Identification

We used IO format here. Unlike original SciBERT repo, we only use a simple linear layer on top of token embeddings.

On test set, we got:

1. **F1 score**: 0.44 
2. **Precision**: 0.43
3. **Recall**: 0.43
4. **Support**: 921

## Credits
1. BERT + BiLSTM + CRF : https://github.com/HandsomeCao/Bert-BiLSTM-CRF-pytorch
2. HuggingFace: https://huggingface.co/transformers/v1.2.0/index.html
3. PyTorch NER: https://github.com/lemonhu/NER-BERT-pytorch
4. BERT: https://github.com/google-research/bert
5. KeyPhrase Extraction : https://github.com/pranav-ust/BERT-keyphrase-extraction
