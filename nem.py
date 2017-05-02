'''
Train and test Neural Event Model (NEM). This module also comes with a main function that acts as a CLI for NEM.
'''

import argparse

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.callbacks import EarlyStopping

from keras_extensions import AnyShapeEmbedding, TimeDistributedRNN, MaskedFlatten
from read_data import DataProcessor


class NEM:
    '''
    Neural Event Model
    '''
    def __init__(self, use_event_structure=True, embedding_dim=50):
        self.use_event_structure = use_event_structure
        self.embedding_dim = embedding_dim
        self.data_processor = DataProcessor()
        self.model = None

    def train_nem(self, inputs, labels, pretrained_embedding_file=None, tune_embedding=False):
        '''
        Train NEM. Depending on whether `use_event_structure` is set in the initializer, the model
        uses either the semantic role structure or just the sentences.
        '''
        pretrained_embedding = None
        if pretrained_embedding_file is not None:
            pretrained_embedding = self.data_processor.get_embedding(pretrained_embedding_file)
        if self.use_event_structure:
            model = self._build_structured_model(inputs, pretrained_embedding, tune_embedding)
        else:
            model = self._build_flat_model(inputs, pretrained_embedding, tune_embedding)
        model.summary()
        model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        early_stopping = EarlyStopping(monitor='val_acc')
        model.fit(inputs, labels, epochs=20, validation_split=0.1, callbacks=[early_stopping])
        self.model = model

    def test_nem(self, inputs, labels):
        '''
        Evaluate NEM on unseen data.
        '''
        metric_values = self.model.evaluate(inputs, labels)
        for metric_name, metric_value in zip(self.model.metrics_names, metric_values):
            print("%s: %.4f" % (metric_name, metric_value))

    def _build_structured_model(self, inputs, pretrained_embedding=None, tune_embedding=False) -> Model:
        # pylint: disable=too-many-locals
        _, num_slots, num_words = inputs.shape
        # (batch_size, num_slots, num_words)
        if pretrained_embedding is None:
            # Override tune_embedding if no pretrained embedding is given.
            tune_embedding = True
        input_layer = Input(shape=(num_slots, num_words), name="EventInput", dtype='int32')
        embedding_weights = None if pretrained_embedding is None else [pretrained_embedding]
        embedding = AnyShapeEmbedding(input_dim=self.data_processor.get_vocabulary_size(),
                                      output_dim=self.embedding_dim, weights=embedding_weights,
                                      mask_zero=True, trainable=tune_embedding, name="Embedding")
        embedded_inputs = embedding(input_layer)  # (batch_size, num_slots, num_words, embedding_dim)
        embedded_inputs = Dropout(0.5)(embedded_inputs)
        encoder = TimeDistributedRNN(LSTM(self.embedding_dim), name="ArgumentEncoder")
        encoded_inputs = encoder(embedded_inputs)  # (batch_size, num_slots, embedding_dim)
        # (batch_size, num_slots * embedding_dim)
        concatenated_slots = MaskedFlatten(name="SlotConcatenator")(encoded_inputs)
        # Note: We essentially have different projection weights for slots here.
        event_composer = Dense(self.embedding_dim, activation='tanh', name="EventComposer")
        # (batch_size, embedding_dim)
        composed_event = event_composer(concatenated_slots)
        # Assuming binary classification.
        event_scorer = Dense(2, activation='softmax', name="EventScorer")
        event_prediction = event_scorer(composed_event)  # (batch_size, 2)
        model = Model(inputs=input_layer, outputs=event_prediction)
        return model

    def _build_flat_model(self, inputs, pretrained_embedding=None, tune_embedding=False) -> Model:
        # pylint: disable=too-many-locals
        _, num_words = inputs.shape
        if pretrained_embedding is None:
            # Override tune_embedding if no pretrained embedding is given.
            tune_embedding = True
        input_layer = Input(shape=(num_words,), name="SentenceInput", dtype='int32')
        embedding_weights = None if pretrained_embedding is None else [pretrained_embedding]
        embedding = Embedding(input_dim=self.data_processor.get_vocabulary_size(), output_dim=self.embedding_dim,
                              weights=embedding_weights, mask_zero=True, trainable=tune_embedding,
                              name="Embedding")
        embedded_inputs = embedding(input_layer)  # (batch_size, num_words, embedding_dim)
        embedded_inputs = Dropout(0.5)(embedded_inputs)
        encoder = LSTM(self.embedding_dim, name="SentenceEncoder")
        encoded_inputs = encoder(embedded_inputs)  # (batch_size, embedding_dim)
        # Project encoding to make the depth of this variant comparable to that of the structured variant.
        # (batch_size, embedding_dim)
        projected_encoding = Dense(self.embedding_dim, activation="tanh", name="Projection")(encoded_inputs)
        sentence_scorer = Dense(2, activation='softmax', name="SentenceScorer")
        sentence_prediction = sentence_scorer(projected_encoding)
        model = Model(inputs=input_layer, outputs=sentence_prediction)
        return model

    def make_inputs(self, filename: str, for_test=False, pad_info=None, include_sentences_in_events=False):
        '''
        Read in a file and use the data processor to make train or test inputs.
        '''
        add_new_words = not for_test
        sentence_inputs, event_inputs, labels = self.data_processor.index_data(filename, add_new_words, pad_info,
                                                                               include_sentences_in_events)
        if self.use_event_structure:
            return event_inputs, labels
        else:
            return sentence_inputs, labels


def main():
    '''
    CLI for NEM
    '''
    argument_parser = argparse.ArgumentParser(description="CLI for training and testing Neural Event Model (NEM)")
    argument_parser.add_argument("--train_file", type=str, help="Train file (JSON). Required for training.")
    argument_parser.add_argument("--test_file", type=str, help="Test file (JSON). Required for testing.")
    argument_parser.add_argument("--embedding_file", type=str, help="Gzipped embedding file.")
    argument_parser.add_argument("--tune_embedding", help="Tune embedding if embedding file is provided.",
                                 action='store_true')
    argument_parser.add_argument("--wanted_args", type=str, nargs='+', help="Arguments to use in the event"
                                 " structure", default=['A0', 'A1', 'AM-TMP', 'AM-LOC'])
    argument_parser.add_argument("--ignore_structure", help="Encode sentences instead of events.",
                                 action='store_true')
    argument_parser.add_argument("--include_sentences_in_events", help="Make the whole sentence an additional"
                                 " argument in the event structure.", action='store_true')
    argument_parser.add_argument("--embedding_dim", type=int, help="Dimensionality of the whole network.",
                                 default=50)
    args = argument_parser.parse_args()
    use_event_structure = not args.ignore_structure
    nem = NEM(use_event_structure=use_event_structure, embedding_dim=args.embedding_dim)
    if args.train_file is not None:
        pad_info = {"wanted_args": args.wanted_args}
        train_inputs, train_labels = nem.make_inputs(args.train_file, for_test=False, pad_info=pad_info,
                                                     include_sentences_in_events=args.include_sentences_in_events)
        nem.train_nem(train_inputs, train_labels, args.embedding_file, args.tune_embedding)
    if args.test_file is not None:
        pad_info_after_train = nem.data_processor.get_pad_info()
        test_inputs, test_labels = nem.make_inputs(args.test_file, for_test=True, pad_info=pad_info_after_train,
                                                   include_sentences_in_events=args.include_sentences_in_events)
        nem.test_nem(test_inputs, test_labels)


if __name__ == "__main__":
    main()
