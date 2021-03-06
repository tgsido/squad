# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract pre-computed feature vectors from a PyTorch BERT model."""

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
MAX_CONTEXT_LEN = 400
MAX_QUESTION_LEN = 50
MAX_SEQ_LENGTH = MAX_CONTEXT_LEN + MAX_QUESTION_LEN

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


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


def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
    return examples


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    #parser.add_argument("--data_type", default="None", type=str, required=True)
    """
    parser.add_argument("--input_file", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str, required=True)
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    """

    ## Other parameters
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--layers", default="-1,-2,-3,-4", type=str)
    # default was 128 before, changed it to be 450
    parser.add_argument("--max_seq_length", default=MAX_SEQ_LENGTH, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                            "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for predictions.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help = "local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {} distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))

    layer_indexes = [int(x) for x in args.layers.split(",")]

    ## Additions ##
    bert_model = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=args.do_lower_case)

    ## Addition to read examples from memory ##
    #examples = read_examples(args.input_file)
    def getExamples():
        examples = []

        def padString(str, final_str_length):
            pad_word = "PAD"
            numWords = len(str.split())
            #print("numWords BEFORE: ", numWords)
            for i in range(final_str_length - numWords):
                str += " " + pad_word
            numWords = len(str.split())
            #print("numWords AFTER: ", numWords)
            return str

        record_file = './data/train.npz'
        """
        record_file = None
        print("user specified ",args.data_type)
        if(args.data_type == 'train'):
            record_file = './data/train.npz'
        elif(args.data_type == 'test'):
            record_file = './data/test.npz'
        elif(args.data_type == 'dev'):
            record_file = './data/dev.npz'
        else:
            println("error - did not specify, train,test,or dev")
            return
        """

        use_squad_v2 = True
        train_dataset = SQuAD(record_file, use_squad_v2)
        count = 0
        example_batches = []
        batch = []
        for idx, example in enumerate(train_dataset):
            padded_context = padString(exampple.context,MAX_CONTEXT_LEN)
            padded_question = padString(example.question,MAX_QUESTION_LEN)
            batch.append(
                InputExample(unique_id=example.id, text_a=padded_context, text_b=padded_question))
            if(idx % 10000 == 0 or idx == len(train_dataset) - 1):
                example_batches.append(batch)
                batch = []
        return example_batches


    example_batches = getExamples()
    numBatches = len(example_batches)
    print("numBatches: ", numBatches, " in train dataset")
    numExamples = sum([len(batch) for batch in example_batches])
    print("numExamples total: ", numExamples)

    model = BertModel.from_pretrained(bert_model)
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    for batchIndex, batch in enumerate(example_batches):
        features = convert_examples_to_features(
            examples=batch, seq_length=MAX_SEQ_LENGTH, tokenizer=tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)

    model.eval()
    output_file = "generated_" + args.data_type + "_bert_embeddings"
    all_output_json = collections.OrderedDict()
    with open(output_file, "w", encoding='utf-8') as writer:
        for input_ids, input_mask, example_index in eval_dataloader:
            print("progess: ", example_index.item(), " / ", (numExamples-1))
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)

            encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask, output_all_encoded_layers=False)
            encoder_layers = encoder_layers.squeeze(0)
            print("encoder_layers.size() : ", encoder_layers.size())

            feature = features[example_index.item()]
            unique_id = int(feature.unique_id)

            output_json = collections.OrderedDict()
            output_json["example_index"] = unique_id
            all_embedding_entries = []
            for (i, token) in enumerate(feature.tokens):
                embedding_entry = {}
                embedding_entry["index"] = i
                embedding_entry["token"] = token
                embedding_entry["vector"] = encoder_layers[i].tolist()
                """
                if token == "PA" or token == "##D":
                    embedding_entry["vector"] = torch.zeros(768).tolist()
                else:
                    embedding_entry["vector"] = encoder_layers[i].tolist()
                """
                all_embedding_entries.append(embedding_entry)
            output_json["embedding_entries"] = all_embedding_entries

            all_output_json[args.data_type + "_" + str(unique_id)] = output_json
        writer.write(json.dumps(all_output_json) + "\n")
        print("all done generating embeddings for ", args.data_type)

if __name__ == "__main__":
    main()
