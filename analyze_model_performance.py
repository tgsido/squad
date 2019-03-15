import argparse
import collections
import json
import logging
import math
import os
import random
import sys
from io import open

import numpy as np

class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 paragraph_text,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.paragraph_text = paragraph_text
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        if self.start_position:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s

def read_squad_examples(input_file, all_ids, is_training, version_2_with_negative, evalDev=False):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                if qas_id not in all_ids:
                    continue
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:
                    if version_2_with_negative:
                        is_impossible = qa["is_impossible"]
                    if (len(qa["answers"]) != 1) and (not is_impossible) and (not evalDev):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        """
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer: '%s' vs. '%s'",
                                           actual_text, cleaned_answer_text)
                            continue
                        """
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    paragraph_text=paragraph_text,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)
    return examples


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_file", default=None, type=str, required=True,
                        help="Filename for output")
    parser.add_argument("--input_file", default=None, type=str, required=True,
                        help="Filename for dev json")
    parser.add_argument("--bert_baseline_submission_file", default=None, type=str, required=False,
                        help="")
    parser.add_argument("--bert_bidaf_submission_file", default=None, type=str, required=False,
                        help="")
    parser.add_argument("--bert_dcn_submission_file", default=None, type=str, required=False,
                        help="")
    parser.add_argument("--bert_answer_pointer_submission_file", default=None, type=str, required=False,
                        help="")
    parser.add_argument("--dcn_submission_file", default=None, type=str, required=False,
                        help="")

    args = parser.parse_args()

    submission_files = dict()
    submission_files["bert_baseline"] = args.bert_baseline_submission_file
    submission_files["bert_bidaf"] = args.bert_bidaf_submission_file
    submission_files["bert_dcn"] = args.bert_dcn_submission_file
    submission_files["bert_answerpointer"] = args.bert_answer_pointer_submission_file
    submission_files["dcn"] = args.dcn_submission_file

    all_ids = []
    ids_filename = "./ids_to_evaluate.txt"
    with open(ids_filename) as f:
        for line in f:
            all_ids.append(line[:-1])

    print("all_ids: ", all_ids)

    # bert_baseline -> dict(id,answer)
    all_submission_dict = dict()
    for submission_file_key in submission_files:
        submission_file = submission_files[submission_file_key]
        if submission_file is None:
            continue
        id_to_answer_dict = dict()
        with open(submission_file) as f:
            for line in f:
                if "Id,Predicted" == line:
                    continue
                commaIndex = line.find(",")
                id = line[:commaIndex]
                #print("id: ", id)
                if id not in all_ids:
                    continue
                answer = line[commaIndex + 1:]
                id_to_answer_dict[id] = answer
        all_submission_dict[submission_file_key] = id_to_answer_dict

    examples = read_squad_examples(input_file=args.input_file, all_ids=all_ids, is_training=True, version_2_with_negative=True, evalDev=True)
    print("len(examples): ", len(examples))

    with open(args.output_file, 'w') as f:
        for example in examples:
            print()
            print("------------------------------------")
            print("ID: ", example.qas_id)
            print("CONTEXT: ", example.paragraph_text)
            print()
            print("QUESTION: ", example.question_text)
            print()
            print("TRUE ANSWER: ", example.orig_answer_text)
            print()
            print("is_impossible: ", example.is_impossible)
            print()
            for submission_file_key in all_submission_dict:
                print("********************************")
                print("model: ", submission_file_key)
                id_to_answer_dict = all_submission_dict[submission_file_key]
                if example.qas_id in id_to_answer_dict:
                    answer = id_to_answer_dict[example.qas_id]
                else:
                    answer = "ID Dropped from this dev set"
                print("answer: ", answer)
                print("********************************")
            print("------------------------------------")
            print()

    f.close()

if __name__ == "__main__":
    main()
