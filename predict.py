import argparse
import torch
import os, json
from collections import defaultdict
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.metric as module_metric
import model.model as module_model
from parse_config import ConfigParser
from utils.util_custom import batch_to_device
import model.baseline as module_baseline
from data_loader.predict_data_loaders import PredictDataLoader

torch.multiprocessing.set_sharing_strategy('file_system')

# data_path
data_path = 'test_samples.json'
output_path = 'multiple_outputs.json'

def main(config):
    logger = config.get_logger('test')
    config['data_loader']['data_path'] = data_path
    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data, 'val')
    # TODO: prepare input tensor from test_input

    # build model architecture
    model = config.init_obj('model', module_model, pt_emb=data_loader.vocab)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    data_loader_args = config['data_loader']['args']
    predict = PredictDataLoader(data_loader_args, vocab = data_loader.vocab, device = device)

    answers = {}
    outputs = {}
    outputs['vid'] = "AnotherMissOh18_008_0000"
    outputs['question'] = "What is Chairman holding?"
    outputs['answers'] = ['Chairman is holding a glass.', 'Chairman is holding chopsticks.', 'Chairman is holding a plate.', 'Chairman is holding a spoon.', 'Chairman is holding a cell phone.']
    with torch.no_grad():
        qid = 0
        data = predict.get_item(outputs['vid'], outputs['question'], outputs['answers'])
        output = model(data)
        _, preds = output.max(dim=1)
        outputs['prediction'] = preds[0].item()       
#        for pred_idx in preds:
#            answers[qid] = pred_idx.item()
#            qid = qid + 1
 
         
#    with open(data_path, 'r') as f:
#        outputs = json.load(f)
#    for qid in range(qid):
#        outputs[qid]['predction'] = answers[qid]
    
    with open(output_path, 'w') as f:
        json.dump(outputs, f, indent=4)
    print("Saved answers at {}".format(output_path))

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-t', '--test', default=None, type=str,
                      help='test model name (default: None)')
    config = ConfigParser.from_args(args)
    main(config)
