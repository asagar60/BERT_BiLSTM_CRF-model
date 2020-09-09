"""Generate Key Phrase of the model"""

import argparse
import random
import logging
import os

import numpy as np
import torch

#from pytorch_pretrained_bert import BertForTokenClassification, BertConfig
from transformers import BertForTokenClassification, BertConfig, BertTokenizer
from evaluate import evaluate
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/msra/', help="Directory containing the dataset")
parser.add_argument('--bert_model_dir', default='bert-base-chinese-pytorch', help="Directory containing the BERT model in PyTorch")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--seed', type=int, default=23, help="random seed for initialization")
parser.add_argument('--restore_file', default='best', help="name of the file in `model_dir` containing weights to load")
parser.add_argument('--multi_gpu', default=False, action='store_true', help="Whether to use multiple GPUs if available")
parser.add_argument('--fp16', default=False, action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--test_file', default = 'h1_7.txt', help = 'path to test file' )

def load_test_sentences(bert_model_dir,sentences_file):
    test_sentences = []
    tokenizer = BertTokenizer.from_pretrained(bert_model_dir, do_lower_case=True)
    with open(sentences_file, 'r',encoding='utf-8') as file:
        for cnt,line in enumerate(file):
            tokens = line.split()
            test_sentences.append(tokenizer.convert_tokens_to_ids(tokens))
    return test_sentences

def yield_data_batch(test_sentences ,params):

    for i in range(len(test_sentences)//params.batch_size):
        # fetch sentences and tags
        sentences = test_sentences[i*params.batch_size:(i+1)*params.batch_size]
        # batch length
        batch_len = len(sentences)
        # compute length of longest sentence in batch
        batch_max_len = max([len(s) for s in sentences])
        max_len = min(batch_max_len, params.max_len)
        # prepare a numpy array with the data, initialising the data with pad_idx
        batch_data = 0 * np.ones((batch_len, max_len))
        # copy the data to the numpy array
        for j in range(batch_len):
            cur_len = len(sentences[j])
            if cur_len <= max_len:
                batch_data[j][:cur_len] = sentences[j]
            else:
                batch_data[j] = sentences[j][:max_len]

        # since all data are indices, we convert them to torch LongTensors
        batch_data = torch.tensor(batch_data, dtype=torch.long)
        # shift tensors to GPU if available
        batch_data = batch_data.to(params.device)
        yield batch_data

def predict(model, data_iterator, params, sentences_file):

    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()
    pred_words = []
    pred_pos = []

    for _ in range(params.eval_steps):
        # fetch the next evaluation batch
        batch_data= next(data_iterator)
        batch_masks = batch_data.gt(0)

        batch_output = model(batch_data, token_type_ids=None, attention_mask=batch_masks)  # shape: (batch_size, max_len, num_labels)
        batch_output = batch_output[0].detach().cpu()

        batch_masks = batch_masks.detach().cpu().numpy().astype('uint8')
        _, indices =  torch.max(batch_output,2)
        for i,idx in enumerate(indices.detach().numpy()):
            #batch_predict.append(batch_data[i,idx==1 and batch_masks[i,:] == True].detach().cpu().numpy())
            pred_pos.append([a and b for a,b in zip(idx, batch_masks[i])])

    output = []
    with open(sentences_file, 'r',encoding='utf-8') as file:
        for cnt,(line,p) in enumerate(zip(file, pred_pos)):
            line = line.split()
            out = [line[i] for i in range(len(line)) if p[i]>0]
            if out:
                output.extend(out)
    print(output)
    with open('output.txt', 'w') as f:
        for out in output:
            f.write("%s\n" % out)
        print('output flused to disk')

if __name__ == '__main__':
    args = parser.parse_args()

    # Load the parameters from json file
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Use GPUs if available
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params.n_gpu = torch.cuda.device_count()
    params.multi_gpu = args.multi_gpu

    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if params.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)  # set random seed for all GPUs
    params.seed = args.seed

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'test.log'))

    test_sentences = load_test_sentences(args.bert_model_dir,args.test_file)
    # Create the input data pipeline
    logging.info("Loading Sentence file...")

    # Specify the test set size
    params.test_size = len(test_sentences)
    params.eval_steps = params.test_size // params.batch_size

    logging.info("- done.")

    # Define the model
    config_path = os.path.join(args.bert_model_dir, 'config.json')
    config = BertConfig.from_json_file(config_path)

    #update config with num_labels
    config.update({"num_labels":2})
    model = BertForTokenClassification(config)
    #model = BertForTokenClassification(config, num_labels=2)

    model.to(params.device)
    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)
    if args.fp16:
        model.half()
    if params.n_gpu > 1 and args.multi_gpu:
        model = torch.nn.DataParallel(model)

    logging.info("Starting evaluation...")
    predict(model = model, data_iterator = yield_data_batch(test_sentences ,params), params = params, sentences_file=args.test_file)
