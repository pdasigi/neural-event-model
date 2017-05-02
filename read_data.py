'''
Process data and prepare inputs for Neural Event Model.
'''
import json
import gzip
from typing import List

import numpy


class DataProcessor:
    '''
    Read in data in json format, index and vectorize words, preparing data for train or test.
    '''
    def __init__(self):
        # All types of arguments seen by the processor. A0, A1, etc.
        self.arg_types = []
        self.max_sentence_length = None
        self.max_arg_length = None
        self.word_index = {"NONE": 0, "UNK": 1}  # None is padding, UNK is OOV.

    def index_data(self, filename, add_new_words=True, pad_info=None, include_sentences_in_events=False):
        '''
        Read data from file, and return indexed inputs. If this is for test, do not add new words to the
        vocabulary (treat them as unk). pad_info is applicable when we want to pad data to a pre-specified
        length (for example when testing, we want to make the sequences the same length as those from train).
        '''
        data = json.load(open(filename))
        indexed_data = []
        for datum in data:
            indexed_sentence = self._index_string(datum["sentence"], add_new_words=add_new_words)
            indexed_event_args = {key: self._index_string(datum["event_structure"][key],
                                                          add_new_words=add_new_words) for key in
                                  datum["event_structure"].keys()}
            if include_sentences_in_events:
                indexed_event_args["sentence"] = indexed_sentence
            indexed_data.append((indexed_sentence, indexed_event_args, datum["label"]))
        sentence_inputs, event_inputs, labels = self.pad_data(indexed_data, pad_info)
        return sentence_inputs, event_inputs, self._make_one_hot(labels)

    def _index_string(self, string: str, add_new_words=True):
        # Assuming the string is already tokenized (with tokens separated by spaces).
        tokens = string.split()
        for token in tokens:
            if token not in self.word_index and add_new_words:
                self.word_index[token] = len(self.word_index)
        token_indices = [self.word_index[token] if token in self.word_index else self.word_index["UNK"] for token
                         in tokens]
        return token_indices

    @staticmethod
    def _make_one_hot(label_indices):
        '''
        Takes integer indices and converts them into one hot representations.
        '''
        output_size = (len(label_indices), numpy.max(label_indices)+1)
        output = numpy.zeros(output_size)
        output[numpy.arange(len(label_indices)), label_indices] = 1
        return output

    def pad_data(self, indexed_data, pad_info):
        '''
        Takes a list of tuples containing indexed sentences, indexed event structures and labels, and returns numpy
        arrays.
        '''
        sentence_inputs = []
        # Setting max sentence length
        if not pad_info:
            pad_info = {}
        indexed_sentences, indexed_event_structures, labels = zip(*indexed_data)
        event_structures_have_sentences = False
        if "sentence" in indexed_event_structures[0]:
            # This means index_data included sentences in event structures. We need to pad accordingly.
            event_structures_have_sentences = True
        if "max_sentence_length" in pad_info:
            self.max_sentence_length = pad_info["max_sentence_length"]
        else:
            self.max_sentence_length = max([len(indexed_sentence) for indexed_sentence in indexed_sentences])
        # Padding and/or truncating sentences
        for indexed_sentence in indexed_sentences:
            sentence_inputs.append(self._pad_indexed_string(indexed_sentence, self.max_sentence_length))

        # Removing unnecessary arguments.
        if "wanted_args" in pad_info:
            self.arg_types = list(pad_info["wanted_args"])
            if "V" not in self.arg_types:
                self.arg_types = ["V"] + self.arg_types
            if "sentence" not in self.arg_types and event_structures_have_sentences:
                self.arg_types += ["sentence"]
        else:
            arg_types = []
            for event_structure in indexed_event_structures:
                arg_types += event_structure.keys()
            self.arg_types = list(set(arg_types))
        # Making ordered event argument indices, converting argument dicts into lists with a canonical order.
        ordered_event_structures = []
        for event_structure in indexed_event_structures:
            ordered_event_structure = [event_structure[arg_type] if arg_type in event_structure else
                                       [self.word_index["NONE"]] for arg_type in self.arg_types]
            ordered_event_structures.append(ordered_event_structure)
        if "max_arg_length" in pad_info:
            self.max_arg_length = pad_info["max_arg_length"]
        else:
            self.max_arg_length = max([max(
                [len(arg) for arg in structure]) for structure in ordered_event_structures])
        event_inputs = []
        for event_structure in ordered_event_structures:
            event_inputs.append([self._pad_indexed_string(indexed_arg, self.max_arg_length) for indexed_arg in
                                 event_structure])
        return numpy.asarray(sentence_inputs), numpy.asarray(event_inputs), numpy.asarray(labels)

    def _pad_indexed_string(self, indexed_string: List[int], max_string_length: int):
        '''
        Pad and/or truncate an indexed string to the max length. Both padding and truncation happen from the left.
        '''
        string_length = len(indexed_string)
        # Padding on or truncating from the left
        padded_string = ([self.word_index["NONE"]] * (max_string_length - string_length) +
                         indexed_string)[-max_string_length:]
        return padded_string

    def get_pad_info(self):
        '''
        Returns the information required to pad or truncate new datasets to make new inputs look like those
        processed so far. This is useful to make test data the same size as train data.
        '''
        pad_info = {}
        if self.arg_types is not None:
            pad_info["wanted_args"] = self.arg_types
        if self.max_arg_length is not None:
            pad_info["max_arg_length"] = self.max_arg_length
        if self.max_sentence_length is not None:
            pad_info["max_sentence_length"] = self.max_sentence_length
        return pad_info

    def get_embedding(self, embedding_file):
        '''
        Reads in a gzipped pretrained embedding file, and returns a numpy array with vectors for words in word
        index.
        '''
        pretrained_embedding = {}
        for line in gzip.open(embedding_file):
            parts = line.strip().split()
            if len(parts) == 2:
                continue
            word = parts[0]
            vector = [float(val) for val in parts[1:]]
            pretrained_embedding[word] = vector
        embedding_size = len(vector)
        embedding = numpy.random.rand(len(self.word_index), embedding_size)
        for word in self.word_index:
            if word in pretrained_embedding:
                embedding[self.word_index[word]] = numpy.asarray(pretrained_embedding[word])
        return embedding

    def get_vocabulary_size(self):
        '''
        Returns the number of unique words seen in indexed data.
        '''
        return len(self.word_index)
