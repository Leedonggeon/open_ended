import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_model
from parse_config import ConfigParser
from utils.util import beam_search, sample_sequence
import os, json

from data_loader.predict_data_loaders import *

from transformers import *
from model.model_gpt import VideoGPT2LMHeadModel

SPECIAL_TOKENS = ["<bos>", "<eos>", "<que>", "<ans>", "<speaker>", "<subtitle>",
                  "<bounding_feature>", "<person>", "<behavior>", "<emotion>", "<video>", "<pad>"]
SPECIAL_TOKENS_DICT = {'bos_token': "<bos>", 'eos_token': "<eos>", 'additional_special_tokens': ["<que>", "<ans>", "<speaker>", "<subtitle>", "<bounding_feature>", "<person>", "<behavior>", "<emotion>", "<video>"], 'pad_token': "<pad>"}

torch.multiprocessing.set_sharing_strategy('file_system')

data_path = 'input_samples.json'
output_path = 'predict_outputs.json'

def main(config):
    logger = config.get_logger('test')
    is_bbfts = config['data_loader']['args']['bbfts']
    args = config['data_loader']['args']

    #### Loading Model ####
    model_checkpoint = config.resume
    checkpoint = str(model_checkpoint).split('/', -1)
    checkpoint.pop()
    tokenizer_checkpoint = ''
    for token in checkpoint:
        tokenizer_checkpoint = tokenizer_checkpoint + token + '/'

    tokenizer_class = GPT2Tokenizer
    tokenizer = tokenizer_class.from_pretrained(tokenizer_checkpoint)
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)

    model_checkpoint = torch.load(model_checkpoint)
    model_class = VideoGPT2LMHeadModel

    model = model_class.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))

    state_dict = model_checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.resize_token_embeddings(len(tokenizer))

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    with open('input_samples.json', 'r') as f:
        input_sample = json.load(f)

    #### Data Loader ####
    predict = PredictDataLoader(args, tokenizer = tokenizer)
 
    vid = input_sample[0]['vid']
    que = input_sample[0]['que']
    
    data = []
    
    with torch.no_grad():
        input_ids, token_type_ids = predict.get_item(vid, que)
        output = {}
        pred = sample_sequence(model, input_ids, token_type_ids, tokenizer, device)
        output['Question'] = que
        output['Prediction'] = tokenizer.decode(pred, skip_special_tokens=True)
        data.append(output)
        with open(output_path, 'w') as outfile:
            json.dump(data, outfile, indent = 4)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
