import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import util
import os

from args import get_example_args
from collections import OrderedDict
from json import dumps
from models import BiDAF
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD
from collections import Counter

import pickle

MAX_CONTEXT_LEN = 400
MAX_QUESTION_LEN = 50

def main(args):

    def url_to_data_path(url):
        return os.path.join('./data/', url.split('/')[-1])

    args.train_file = url_to_data_path(args.train_url)
    args.dev_file = url_to_data_path(args.dev_url)
    if args.include_test_examples:
        args.test_file = url_to_data_path(args.test_url)
    glove_dir = url_to_data_path(args.glove_url.replace('.zip', ''))
    glove_ext = '.txt' if glove_dir.endswith('d') else '.{}d.txt'.format(args.glove_dim)
    args.glove_file = os.path.join(glove_dir, os.path.basename(glove_dir) + glove_ext)

    pickle_in = open("idx2word_dict.pickle","rb")
    idx2token_dict = pickle.load(pickle_in)
    #print("idx2token_dict: ", idx2token_dict.items())

    def getStringFromIdxs(idxs):
        print("idxs: " , idxs.tolist())
        idxs = idxs.tolist()
        str = ""
        for idx in idxs:
            str += idx2token_dict[idx] + " "
        return str

    def padString(str, final_str_length):
        pad_word = "PAD"
        numWords = len(str.split())
        print("numWords BEFORE: ", numWords)
        for i in range(final_str_length - numWords - 1):
            str += " " + pad_word
        numWords = len(str.split())
        print("numWords AFTER: ", numWords)
        return str

    train_dataset = SQuAD(args.train_record_file, args.use_squad_v2)
    count = 0
    for context, question, cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in train_dataset:
        print("------------")
        print("count: ", count)
        print("len(cw_idxs) : ", len(cw_idxs))
        print("len(qw_idxs) : ", len(qw_idxs))
        print("context: ", context)
        print("question: ", question)
        padded_context = padString(context,len(cw_idxs))
        padded_question = padString(question,len(qw_idxs))
        print("padded_context: ", padded_context)
        print("padded_question: ", padded_question)
        """
        rebuiltContextStr = getStringFromIdxs(cw_idxs)
        rebuiltQuestionStr = getStringFromIdxs(qw_idxs)
        print("rebuiltContextStr: ", rebuiltContextStr)
        print("rebuiltQuestionStr: ", rebuiltQuestionStr)
        """

        print("------------")
        count += 1
        if count == 2:
            break


if __name__ == '__main__':
    main(get_example_args())
