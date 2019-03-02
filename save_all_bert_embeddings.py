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
import os

"""
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
"""
class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0      0   0    1  1  1   1  1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        """
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (example.unique_id))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))
        """
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def save_bert_embeddings(data_type, max_context_len=400, max_question_len=50):
    directory = "/datasquad/" + data_type + "_bert_embeddings"
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        print("dir already exits: ", directory)

    ## SET DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    MAX_CONTEXT_LEN = max_context_len
    MAX_QUESTION_LEN = max_question_len
    MAX_SEQ_LENGTH = MAX_CONTEXT_LEN + MAX_QUESTION_LEN

    #ids = ids.tolist()
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
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)

    model.eval()

    for input_ids, input_mask, example_index in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)

        save_dir = directory + "/"  + str(example_index.item()) + '.pt'
        if not os.path.exists(save_dir):
            with torch.no_grad():
                encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask, output_all_encoded_layers=False)
                embeddings = encoder_layers
                print("example_index: ", example_index.item())
                torch.save(embeddings, save_dir)

    print("all done saving embeddings for ", data_type)

data_types = ['dev','test','train']
for data_type in data_types:
    save_bert_embeddings(data_type)
