'''
Train and test Neural Event Model (NEM). This module also comes with a main function that acts as a CLI for NEM.
'''

# pylint: disable=wrong-import-position
import sys
import argparse
import pickle
import os
import numpy
numpy.random.seed(21957)

from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Lambda, merge

from metrics import precision, recall, f1_score
from keras_extensions import AnyShapeEmbedding, TimeDistributedRNN, MaskedFlatten
from read_data import DataProcessor

from onto_lstm.encoders import OntoLSTMEncoder

NUM_EPOCHS = 50
PATIENCE = 5

class NEM:
    '''
    Neural Event Model
    '''
    def __init__(self, use_event_structure=True, embedding_dim=50,
                 onto_aware=False, num_senses=3, num_hyps=5):
        self.use_event_structure = use_event_structure
        self.embedding_dim = embedding_dim
        self.data_processor = DataProcessor(num_senses=num_senses, num_hyps=num_hyps)
        self.model = None
        model_type = "structured" if use_event_structure else "flat"
        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")
        self.model_prefix = "saved_models/nem_%s_dim=%d" % (model_type, embedding_dim)
        # Custom metrics
        self.custom_objects = {"precision": precision, "recall": recall, "f1_score": f1_score}
        if use_event_structure:
            # Custom layers
            self.custom_objects.update({"AnyShapeEmbedding": AnyShapeEmbedding,
                                        "MaskedFlatten": MaskedFlatten,
                                        "TimeDistributedRNN": TimeDistributedRNN})
        self._onto_aware = onto_aware
        self.num_hyps = num_hyps
        self.num_senses = num_senses
        if self._onto_aware:
            # Declaring ontolstm only to update custom objects.
            onto_lstm = OntoLSTMEncoder(num_senses, num_hyps, True, True)
            self.custom_objects.update(onto_lstm.get_custom_objects())

    def train_nem(self, inputs, labels, pretrained_embedding_file=None, tune_embedding=False):
        '''
        Train NEM. Depending on whether `use_event_structure` is set in the initializer, the model
        uses either the semantic role structure or just the sentences.
        '''
        if self.use_event_structure:
            model = self._build_structured_model(inputs=inputs, pretrained_embedding_file=pretrained_embedding_file,
                                                 tune_embedding=tune_embedding)
        else:
            model = self._build_flat_model(inputs=inputs, pretrained_embedding_file=pretrained_embedding_file,
                                           tune_embedding=tune_embedding)
        model.summary()
        model.compile("adam", "categorical_crossentropy", metrics=["accuracy", precision,
                                                                   recall, f1_score])
        self.model = model
        best_accuracy = 0.0
        best_epoch = 0
        num_worse_epochs = 0
        for i in range(NUM_EPOCHS):
            print("Epoch %d" % i, file=sys.stdout)
            history = self.model.fit(inputs, labels, nb_epoch=1, validation_split=0.1)
            validation_accuracy = history.history['val_acc'][0]
            if validation_accuracy > best_accuracy:
                self._save_model(i)
                best_epoch = i
                num_worse_epochs = 0
                best_accuracy = validation_accuracy
            elif validation_accuracy < best_accuracy:
                num_worse_epochs += 1
                if num_worse_epochs >= PATIENCE:
                    print("Ran out of patience. Stopping training.", file=sys.stdout)
                    break
        self._save_model_as_best(best_epoch)

    def test_nem(self, inputs, labels, output_filename=None):
        '''
        Evaluate NEM on unseen data.
        '''
        metric_values = self.model.evaluate(inputs, labels)
        for metric_name, metric_value in zip(self.model.metrics_names, metric_values):
            print("%s: %.4f" % (metric_name, metric_value))
        if output_filename is not None:
            predictions = self.model.predict(inputs)
            predicted_classes = numpy.argmax(predictions, axis=-1)
            output_file = open(output_filename, "w")
            for pred_class in predicted_classes:
                print(pred_class, file=output_file)

    def _build_structured_model(self, inputs, pretrained_embedding_file=None, tune_embedding=False) -> Model:
        # pylint: disable=too-many-locals
        embedding_weights = None
        if pretrained_embedding_file is None:
            # Override tune_embedding if no pretrained embedding is given.
            tune_embedding = True
        else:
            pretrained_embedding = self.data_processor.get_embedding(pretrained_embedding_file)
            embedding_weights = [pretrained_embedding]
        num_slots = inputs.shape[1]
        input_layer = Input(shape=inputs.shape[1:], name="EventInput", dtype='int32')
        if self._onto_aware:
            self.onto_lstm = OntoLSTMEncoder(num_senses=self.num_senses,
                                             num_hyps=self.num_hyps,
                                             use_attention=True,
                                             set_sense_priors=True,
                                             data_processor=self.data_processor.onto_lstm_data_processor,
                                             embed_dim=self.embedding_dim,
                                             return_sequences=False,
                                             tune_embedding=tune_embedding)
            all_slot_encodings = []
            for i in range(num_slots):
                slot_input_layer = Lambda(lambda x: x[:, i], output_shape=(1,) + inputs.shape[2:])
                slot_input = slot_input_layer(input_layer)
                slot_encoding = self.onto_lstm.get_encoded_phrase(phrase_input_layer=slot_input,
                                                                  embedding=pretrained_embedding_file,
                                                                  dropout={"embedding": 0.5, "encoder": 0.2})
                all_slot_encodings.append(slot_encoding)
            concatenated_slots = merge(all_slot_encodings, mode='concat', concat_axis=1)

        else:
            embedding = AnyShapeEmbedding(input_dim=self.data_processor.get_vocabulary_size(),
                                          output_dim=self.embedding_dim, weights=embedding_weights,
                                          mask_zero=True, trainable=tune_embedding, name="Embedding")
            embedded_inputs = embedding(input_layer)  # (batch_size, num_slots, num_words, embedding_dim)
            embedded_inputs = Dropout(0.5)(embedded_inputs)
            encoder = TimeDistributedRNN(LSTM(self.embedding_dim), name="ArgumentEncoder")
            encoded_inputs = encoder(embedded_inputs)  # (batch_size, num_slots, embedding_dim)
            encoded_inputs = Dropout(0.2)(encoded_inputs)
            # (batch_size, num_slots * embedding_dim)
            concatenated_slots = MaskedFlatten(name="SlotConcatenator")(encoded_inputs)
        # Note: We essentially have different projection weights for slots here.
        event_composer = Dense(self.embedding_dim, activation='tanh', name="EventComposer")
        # (batch_size, embedding_dim)
        composed_event = event_composer(concatenated_slots)
        # Assuming binary classification.
        event_scorer = Dense(2, activation='softmax', name="EventScorer")
        event_prediction = event_scorer(composed_event)  # (batch_size, 2)
        model = Model(input=input_layer, output=event_prediction)
        return model

    def _build_flat_model(self, inputs, pretrained_embedding_file=None, tune_embedding=False) -> Model:
        # pylint: disable=too-many-locals
        input_layer = Input(shape=inputs.shape[1:], name="SentenceInput", dtype='int32')
        if self._onto_aware:
            self.onto_lstm = OntoLSTMEncoder(num_senses=self.num_senses,
                                             num_hyps=self.num_hyps,
                                             use_attention=True,
                                             set_sense_priors=True,
                                             data_processor=self.data_processor.onto_lstm_data_processor,
                                             embed_dim=self.embedding_dim,
                                             return_sequences=False,
                                             tune_embedding=tune_embedding)
            encoded_inputs = self.onto_lstm.get_encoded_phrase(phrase_input_layer=input_layer,
                                                               embedding=pretrained_embedding_file,
                                                               dropout={"embedding": 0.5, "encoder": 0.2})
        else:
            embedding_weights = None
            if pretrained_embedding_file is None:
                # Override tune_embedding if no pretrained embedding is given.
                tune_embedding = True
            else:
                pretrained_embedding = self.data_processor.get_embedding(pretrained_embedding_file)
                embedding_weights = [pretrained_embedding]
            embedding = Embedding(input_dim=self.data_processor.get_vocabulary_size(), output_dim=self.embedding_dim,
                                  weights=embedding_weights, mask_zero=True, trainable=tune_embedding,
                                  name="Embedding")
            embedded_inputs = embedding(input_layer)  # (batch_size, num_words, embedding_dim)
            embedded_inputs = Dropout(0.5)(embedded_inputs)
            encoder = LSTM(self.embedding_dim, name="SentenceEncoder")
            encoded_inputs = encoder(embedded_inputs)  # (batch_size, embedding_dim)
            encoded_inputs = Dropout(0.2)(encoded_inputs)
        # Project encoding to make the depth of this variant comparable to that of the structured variant.
        # (batch_size, embedding_dim)
        projected_encoding = Dense(self.embedding_dim, activation="tanh", name="Projection")(encoded_inputs)
        sentence_scorer = Dense(2, activation='softmax', name="SentenceScorer")
        sentence_prediction = sentence_scorer(projected_encoding)
        model = Model(input=input_layer, output=sentence_prediction)
        return model

    def make_inputs(self, filename: str, for_test=False, pad_info=None, include_sentences_in_events=False):
        '''
        Read in a file and use the data processor to make train or test inputs.
        '''
        add_new_words = not for_test
        sentence_inputs, event_inputs, labels = self.data_processor.index_data(filename, add_new_words, pad_info,
                                                                               include_sentences_in_events,
                                                                               self._onto_aware)
        if self.use_event_structure:
            return event_inputs, labels
        else:
            return sentence_inputs, labels

    def _save_model(self, epoch: int):
        model_file = "%s_%d.h5" % (self.model_prefix, epoch)
        data_processor_file = "%s_dp.pkl" % self.model_prefix
        self.model.save(model_file)
        pickle.dump(self.data_processor, open(data_processor_file, "wb"))

    def _save_model_as_best(self, epoch: int):
        best_model_file = "%s_%d.h5" % (self.model_prefix, epoch)
        new_name = "%s_best.h5" % self.model_prefix
        os.rename(best_model_file, new_name)

    def load_model(self, epoch: int=None):
        '''
        Load a pretrained model, optionally from a specific epoch. If no epoch is specified, the model that gave
        the best validation accuracy will be loaded.
        '''
        data_processor_file = "%s_dp.pkl" % self.model_prefix
        self.data_processor = pickle.load(open(data_processor_file, "rb"))
        if epoch is None:
            model_file = "%s_best.h5" % self.model_prefix
        else:
            model_file = "%s_%d.h5" % (self.model_prefix, epoch)
        # If the following line throws errors, it may be because the version of Keras is too old to
        # support deserializing Lambda layers in the model. The fix is the following
        # https://github.com/keras-team/keras/pull/8592/files#diff-56dc3cc42e1732fdb3a3c2c3c8efa32a
        self.model = load_model(model_file, custom_objects=self.custom_objects)


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
                                 " structure")
    argument_parser.add_argument("--ignore_structure", help="Encode sentences instead of events.",
                                 action='store_true')
    argument_parser.add_argument("--onto_aware", help="Use OntoLSTM as the encoder",
                                 action='store_true')
    argument_parser.add_argument("--num_senses", type=int, default=3,
                                 help="Number of senses if using OntoLSTM (default 3)")
    argument_parser.add_argument("--num_hyps", type=int, default=5,
                                 help="Number of hypernyms if using OntoLSTM (default 5)")
    argument_parser.add_argument("--include_sentences_in_events", help="Make the whole sentence an additional"
                                 " argument in the event structure.", action='store_true')
    argument_parser.add_argument("--embedding_dim", type=int, help="Dimensionality of the whole network.",
                                 default=50)
    argument_parser.add_argument("--output_file", type=str, help="Output file name to print predictions.")
    args = argument_parser.parse_args()
    use_event_structure = not args.ignore_structure
    nem = NEM(use_event_structure=use_event_structure, embedding_dim=args.embedding_dim,
              onto_aware=args.onto_aware, num_senses=args.num_senses, num_hyps=args.num_hyps)
    if args.train_file is not None:
        pad_info = {"wanted_args": args.wanted_args} if args.wanted_args is not None else {}
        train_inputs, train_labels = nem.make_inputs(args.train_file, for_test=False, pad_info=pad_info,
                                                     include_sentences_in_events=args.include_sentences_in_events)
        nem.train_nem(inputs=train_inputs, labels=train_labels, pretrained_embedding_file=args.embedding_file,
                      tune_embedding=args.tune_embedding)
    if args.test_file is not None:
        # Even if we trained NEM in this run, we should load the best model.
        nem.load_model()
        nem.model.summary()
        pad_info_after_train = nem.data_processor.get_pad_info()
        test_inputs, test_labels = nem.make_inputs(args.test_file, for_test=True, pad_info=pad_info_after_train,
                                                   include_sentences_in_events=args.include_sentences_in_events)
        nem.test_nem(test_inputs, test_labels, output_filename=args.output_file)


if __name__ == "__main__":
    main()
