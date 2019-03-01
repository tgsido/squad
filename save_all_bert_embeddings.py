from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import logging
import json
import re

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
## Additions ##
from util import collate_fn, SQuAD

"""
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
"""


def save_bert_embeddings(data_type, max_context_len=400, max_question_len=50):
    ## SET DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    MAX_CONTEXT_LEN = max_context_len
    MAX_QUESTION_LEN = max_question_len
    MAX_SEQ_LENGTH = MAX_CONTEXT_LEN + MAX_QUESTION_LEN

    ids = ids.tolist()
    #print("ids: ", ids)
    #print("len(ids): ", len(ids))

    ## Uncomment to test quickly ##
    #return torch.ones((len(ids), MAX_SEQ_LENGTH, 768), device=device)




    def padString(str, final_str_length):
        pad_word = "PAD"
        numWords = len(str.split())
        #print("numWords BEFORE: ", numWords)
        for i in range(final_str_length - numWords):
            str += " " + pad_word
        numWords = len(str.split())
        #print("numWords AFTER: ", numWords)
        return str

    ## BERT MODEL TYPE AND TOKENIZER ##
    bert_model = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(bert_model)


    ## DATAFILE ##
    record_file = None
    print("user specified ",data_type)
    if(data_type == 'train'):
        record_file = './data/train.npz'
    elif(data_type == 'test'):
        record_file = './data/test.npz'
    elif(data_type == 'dev'):
        record_file = './data/dev.npz'

    use_squad_v2 = True

    ## GET DATASET ##
    dataset = SQuAD(record_file, use_squad_v2)

    ## GET ALL EXAMPLES ##
    examples = []
    count = 0

    print("dataset ids len: ", len(dataset))
    for context, question, cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, id in dataset:
        padded_context = padString(context,MAX_CONTEXT_LEN)
        padded_question = padString(question,MAX_QUESTION_LEN)
        examples.append(
            InputExample(unique_id=id, text_a=padded_context, text_b=padded_question))
        count += 1

    numExamples = len(examples)
    print("numExamples: ", numExamples, " in ", data_type, " dataset")

    features = convert_examples_to_features(
        examples=examples, seq_length=MAX_SEQ_LENGTH, tokenizer=tokenizer)

    model = BertModel.from_pretrained(bert_model)
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    ## CREATE TENSORDATASET ##
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)

    ## CREATE DATALOADER ##
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=numExamples)

    model.eval()

    embeddings = torch.zeros((numExamples, MAX_SEQ_LENGTH, 768), device=device)
    for input_ids, input_mask, example_index in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        with torch.no_grad():
            encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask, output_all_encoded_layers=False)
            embeddings = encoder_layers
        print("encoder_layers.size() : ", encoder_layers.size())


    print("all done generating embeddings for ", data_type)
    print("embeddings.size()", embeddings.size())
    torch.save(embeddings, data_type + '_bert_embeddings.pt')
    print("all done saving embeddings for ", data_type)

data_types = ['train','dev','test']
for data_type in data_types:
    save_bert_embeddings(data_type)
