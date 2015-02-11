import sys
import numpy, theano, cPickle
from nem_clean import NEM
from read_data import vectorize_sentences, vectorize_events
from random import shuffle

def run_nem(wordvecs, lr=0.001, training_epochs=100, train_sents_file="./train_sents.txt", sup_train_events_file='./supervised_train_events.txt', label_file="./labels.txt", valid_prop=0.2, test_events_file="./test_events.txt", test_output = './test_output.txt', activation='sigmoid', prev_sharedvector_file='', prev_wordindex_file=''):
	args = ['V', 'A0', 'A1', 'AM-TMP', 'AM-LOC']
	labels = [int(x.strip()) for x in open(label_file)]
	"""	
	if prev_sharedvector_file != '' and prev_wordindex_file != '':
		print >>sys.stderr, "Loading trained word vectors"
		# Make prev_sharedvectors a list so as to be able to append to it.
		prev_sharedvectors = list(cPickle.load(open(prev_sharedvector_file, "rb")))
		prev_wordindex = {line.split()[0]: int(line.strip().split()[1]) for line in open(prev_wordindex_file)}
		#(train_valid_events, train_valid_splitsizes), train_valid_sentencereps, (tr_wordindex, tr_sharedvectors) = vectorize_data(wordvecs, train_file, args, wordindex=prev_wordindex, sharedvectors=prev_sharedvectors)
		train_valid_sentencereps, (tr_wordindex, tr_sharedvectors) = vectorize_sentences(wordvecs, train_sents_file, wordindex=prev_wordindex, sharedvectors=prev_sharedvectors)
	else:
		#(train_valid_events, train_valid_splitsizes), train_valid_sentencereps, (tr_wordindex, tr_sharedvectors) = vectorize_data(train_file, wordvecs, args)
		train_valid_sentencereps, (tr_wordindex, tr_sharedvectors) = vectorize_sentences(train_sents_file, wordvecs)"""


	# Begin temporaty fix
	tr_wordindex = {x.split()[0]:int(x.strip().split()[1]) for x in open('vocab.txt')}
	wordfile="words.lst"
	embedfile="embeddings.txt"
	wordvecs = {word.strip():numpy.asarray([[float(i) for i in vector.split()]], dtype=theano.config.floatX) for word, vector in zip(open(wordfile), open(embedfile))}
	tr_sharedvectors = list(cPickle.load(open('vocab_final.pkl', 'rb')))
	# End temporary fix

	# Continue building indexed_vocab on supervised training data, and test data
	#(sup_train_valid_events, sup_train_valid_splitsizes), sup_train_valid_sentencereps, (str_wordindex, str_sharedvectors) = vectorize_data(sup_train_file, wordvecs, args, wordindex=tr_wordindex, sharedvectors=tr_sharedvectors)
	#sup_train_valid_sentencereps, (str_sent_wordindex, str_sent_sharedvectors) = vectorize_sentences(sup_train_sents_file, wordvecs, wordindex=tr_wordindex, sharedvectors=tr_sharedvectors)
	(sup_train_valid_events, sup_train_valid_splitsizes), (str_wordindex, str_sharedvectors) = vectorize_events(sup_train_events_file, wordvecs, args, wordindex=tr_wordindex, sharedvectors=tr_sharedvectors)
	#(sup_train_events, sup_train_splitsizes, sup_train_sentencereps), sup_train_text, (str_wordindex, str_sharedvectors) = vectorize_data(wordvecs, sup_train_file, args, wordindex=prev_wordindex, sharedvectors=prev_sharedvectors)
	#(test_events, test_splitsizes), _, (wordindex, sharedvectors) = vectorize_data(wordvecs, test_file, args, wordindex=str_wordindex, sharedvectors=str_sharedvectors)
	(test_events, test_splitsizes), (wordindex, sharedvectors) = vectorize_events(test_events_file, wordvecs, args, wordindex=str_wordindex, sharedvectors=str_sharedvectors)
	numpy_vecs = numpy.asarray(sharedvectors, dtype=theano.config.floatX)
	nem = NEM(numpy_vecs, numargs=len(args), lr=lr, activation=activation)

	train_prop = 1 - valid_prop
	# Getting data for phase 1
	"""train_valid_sent_size = len(train_valid_sentencereps)
	train_sent_size = int(train_prop * train_valid_sent_size)
	shuffle(train_valid_sentencereps)
	train_sentencereps = train_valid_sentencereps[:train_sent_size]
	valid_sentencereps = train_valid_sentencereps[train_sent_size:]"""
	# Getting data for phase 2
	train_valid_event_size = len(sup_train_valid_events)
	train_event_size = int(train_prop * train_valid_event_size)
	sup_train_valid_data = zip(sup_train_valid_events, sup_train_valid_splitsizes, labels)
	shuffle(sup_train_valid_data)
	shuffled_sup_train_valid_events, shuffled_sup_train_valid_splitsizes, shuffled_labels = (list(x) for x in zip(*sup_train_valid_data))
	sup_train_events = shuffled_sup_train_valid_events[:train_event_size]
	sup_train_splitsizes = shuffled_sup_train_valid_splitsizes[:train_event_size]
	sup_train_labels = shuffled_labels[:train_event_size]
	sup_valid_events = shuffled_sup_train_valid_events[train_event_size:]
	sup_valid_splitsizes = shuffled_sup_train_valid_splitsizes[train_event_size:]
	sup_valid_labels = shuffled_labels[train_event_size:]
	print >>sys.stderr, "Training model..."
	#logfh = open("./log", 'w', 0)
	"""print >>sys.stderr, "Phase 1"
	for epoch in range(training_epochs):
		trainscores, traincosts = nem.train_arg_composition(train_sentencereps, vocabsize=len(wordindex))
		validscores, validcosts = nem.train_arg_composition(valid_sentencereps, vocabsize=len(wordindex), no_updates=True)
		print >>sys.stderr, "Finished epoch %d, average train score is %f, average train cost is %f"%(epoch+1, sum(trainscores)/len(trainscores), sum(traincosts)/len(traincosts))
		print >>sys.stderr, "\taverage validation score is %f, average validation cost is %f"%(sum(validscores)/len(validscores), sum(validcosts)/len(validcosts))
		if (epoch+1)%10 == 0:
			paramfile = open("argparam"+str(epoch+1)+".pkl", "wb")
			argParam = (nem.argW.get_value(), nem.argbhid.get_value(), nem.argScorer.get_value())		
			cPickle.dump(argParam, paramfile)
			paramfile.close()
			vocabfile = open("vocab"+str(epoch+1)+".pkl", "wb")
			learnedVocab = nem.ts_vocab.get_value()		
			cPickle.dump(learnedVocab, vocabfile)
			vocabfile.close()
	paramfile = open("argparam_final.pkl", "wb")
	argParam = (nem.argW.get_value(), nem.argbhid.get_value(), nem.argScorer.get_value())		
	cPickle.dump(argParam, paramfile)
	paramfile.close()
	vocabfile = open("vocab_final.pkl", "wb")
	learnedVocab = nem.ts_vocab.get_value()		
	cPickle.dump(learnedVocab, vocabfile)
	vocabfile.close()
	print >>sys.stderr, "Done!" """
	#return
	
	# Begin temporary fix
	print >>sys.stderr, "Loading phase 1 parameters"
	argW, argbhid, argScorer = cPickle.load(open("adapt_argparam_supervised_final.pkl", "rb"))
	nem.argW.set_value(argW)
	nem.argbhid.set_value(argbhid)
	nem.argScorer.set_value(argScorer)
	print >>sys.stderr, "Getting all arg embeddings.."
	#testargfile = open("arg_embeds.pkl", "wb")
	# End temporary fix
    
	train_indexrep, train_eventargs, train_argpositions = nem.get_arg_embeddings(sup_train_events, sup_train_splitsizes, args)
	valid_indexrep, valid_eventargs, valid_argpositions = nem.get_arg_embeddings(sup_valid_events, sup_valid_splitsizes, args)
	print >>sys.stderr, "Phase 2"
	for epoch in range(training_epochs):
		trainscores, traincosts = nem.train_event_composition(train_indexrep, train_eventargs, train_argpositions, sup_train_labels)
		validscores, validcosts = nem.train_event_composition(valid_indexrep, valid_eventargs, train_argpositions, sup_valid_labels, no_updates=True)
		print >>sys.stderr, "Finished epoch %d, average train score is %f, average train cost is %f"%(epoch+1, sum(trainscores)/len(trainscores), sum(traincosts)/len(traincosts))
		print >>sys.stderr, "\taverage valid score is %f, average valid cost is %f"%(sum(validscores)/len(validscores), sum(validcosts)/len(validcosts))
		if (epoch+1)%10 == 0:
			paramfile = open("eventparam_supervised"+str(epoch+1)+".pkl", "wb")
			eventParam = (nem.eventW.get_value(), nem.eventbhid.get_value(), nem.eventScorer.get_value())
			cPickle.dump(eventParam, paramfile)
			paramfile.close()	
	paramfile = open("eventparam_supervised_final.pkl", "wb")
	eventParam = (nem.eventW.get_value(), nem.eventbhid.get_value(), nem.eventScorer.get_value())
	cPickle.dump(eventParam, paramfile)
	paramfile.close()	
	"""vocabindexfile = open("vocab.txt", 'w')

	for ind in wordindex:
		print >>vocabindexfile, ind, wordindex[ind]
	vocabindexfile.close()"""
	
	print >>sys.stderr, "Composing test events"
	test_indexrep, test_eventargs, test_argpositions = nem.get_arg_embeddings(test_events, test_splitsizes, args)
	#cPickle.dump((test_argindex, test_indexrep, test_eventargs), testargfile)
	test_scores, test_embeddings = nem.score_event_composition(test_eventargs, test_indexrep)
	outfh = open(test_output, "w")
	for score, embedding in zip(test_scores, test_embeddings):
		print >>outfh, score, embedding
	print >>sys.stderr, "Done!"

def read_word_vectors(wordfile="words.lst", embedfile="embeddings.txt"):
	return {word.strip():numpy.asarray([[float(i) for i in vector.split()]], dtype=theano.config.floatX) for word, vector in zip(open(wordfile), open(embedfile))}

if __name__ == '__main__':
	wordvecs = read_word_vectors()
	print >>sys.stderr, "Read %d word vectors"%(len(wordvecs))
	run_nem(wordvecs, train_sents_file="./train_sents.txt", sup_train_events_file='./supervised_train_events.txt', label_file="./labels.txt", test_events_file="./test_events.txt", test_output = './test_output.txt', activation='sigmoid' )

