# Import parts from nem.py.  Make it clean!
# After cleaning it, put each class in a separate file.  Some modularity, man!

import cPickle
import gzip
import os
import sys
import time
from random import randint, sample

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class ConcreteArgNode(object):
	def __init__(self, outputRep, vocabIndex=None, leftChild=None, rightChild=None, fakeReps=[], fakeIndex=None):
		self.leftChild = leftChild
		self.rightChild = rightChild
		self.outputRep = outputRep
		self.vocabIndex = vocabIndex
		self.fakeReps = fakeReps
		self.fakeIndex = fakeIndex


class AbstractNode(object):
	def __init__(self, ts_vocab, W, bhid, scorer, typ, n_hid, n_vis, reps_av = None, lr=0.1, activation='sigmoid'):
		self.n_hidden = n_hid
		self.n_visible = n_vis
		numpy_rng = numpy.random.RandomState(89677)
		theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
		self.lr = lr
		
		if typ == "event":
			# The parameters for event training (phase 2) are W_event, b_event, S_event and ts_vocab for modifying verb representations
			self.params = [W, bhid, scorer, ts_vocab]
			# TODO: Do not assume there are four arguments.
			verbindex = T.iscalar('verbindex')
			a0, a1, a2, a3  = T.matrices('a0', 'a1', 'a2', 'a3')
			self.input = T.concatenate([ts_vocab[verbindex], a0, a1, a2, a3], axis=1)
			# TODO: Get rid of autoencoder, and the dependency on dA
			#self.autoencoder = dA(numpy_rng=numpy_rng,
			#			theano_rng=theano_rng,
			#			input=self.input,
			#			n_visible=n_vis,
			#			n_hidden=n_hid,
			#			W=W,
			#			bhid=bhid)
			#self.output = self.autoencoder.get_hidden_values(self.input)
			self.output = self.get_hidden_values(self.input, W, bhid, activation)
			label = T.iscalar('label')
			self.score = T.sum(T.nnet.sigmoid(T.dot(self.output, scorer.T)))
			self.cost = -T.dot(label, T.log(self.score)) - T.dot((1 - label), T.log(1 - self.score))
			
			self.scoreInputs = [verbindex, a0, a1, a2, a3]
			self.costInputs = self.scoreInputs + [label]

		else:
			# The parameters for arg training (phase 1) are W_arg, b_arg and S_arg
			self.params = [W, bhid, scorer]
			# We also add ts_vocab to params for parents of leaf nodes
			if reps_av is None:
				leftindex, rightindex, fake_leftindex, fake_rightindex = T.iscalars('leftindex', 'rightindex', 'fake_leftindex', 'fake_rightindex')
				leftrep = ts_vocab[leftindex]
				rightrep = ts_vocab[rightindex]
				fake_leftrep = ts_vocab[fake_leftindex]
				fake_rightrep = ts_vocab[fake_rightindex]
				self.scoreInputs = [leftindex, rightindex]
				fake_scoreInputs = [fake_leftindex, fake_rightindex]
				self.params.append(ts_vocab)
			elif reps_av == 'l':
				rightindex, fake_rightindex = T.iscalars('rightindex', 'fake_rightindex')
				leftrep, fake_leftrep = T.matrices('leftrep', 'fake_leftrep')
				rightrep = ts_vocab[rightindex]
				fake_rightrep = ts_vocab[fake_rightindex]
				self.scoreInputs = [leftrep, rightindex]
				fake_scoreInputs = [fake_leftrep, fake_rightindex]
				self.params.append(ts_vocab)
			elif reps_av == 'r':
				leftindex, fake_leftindex = T.iscalars('leftindex', 'fake_leftindex')
				rightrep, fake_rightrep = T.matrices('rightrep', 'fake_rightrep')
				leftrep = ts_vocab[leftindex]
				fake_leftrep = ts_vocab[fake_leftindex]
				self.scoreInputs = [leftindex, rightrep]
				fake_scoreInputs = [fake_leftindex, fake_rightrep]
				self.params.append(ts_vocab)
			elif reps_av == 'lr':
				leftrep, fake_leftrep = T.matrices('leftrep', 'fake_leftrep')
				rightrep, fake_rightrep = T.matrices('rightrep', 'fake_rightrep')
				self.scoreInputs = [leftrep, rightrep]
				fake_scoreInputs = [fake_leftrep, fake_rightrep]
			self.input = T.concatenate([leftrep, rightrep], axis=1)
			left_fake_input = T.concatenate([fake_leftrep, rightrep], axis=1)
			right_fake_input = T.concatenate([leftrep, fake_rightrep], axis=1)
				
			# TODO: Get rid of autoencoder, and the dependency on dA
			#self.autoencoder = dA(numpy_rng=numpy_rng,
			#			theano_rng=theano_rng,
			#			input=self.input,
			#			n_visible=n_vis,
			#			n_hidden=n_hid,
			#			W=W,
			#			bhid=bhid)
			#fake_outputs = [self.autoencoder.get_hidden_values(left_fake_input), self.autoencoder.get_hidden_values(right_fake_input)]
			fake_outputs = [self.get_hidden_values(left_fake_input, W, bhid, activation), self.get_hidden_values(right_fake_input, W, bhid, activation)]
		
			#self.output = self.autoencoder.get_hidden_values(self.input)
			self.output = self.get_hidden_values(self.input, W, bhid, activation)
			# Summing to make the "matrix" of dimension 1x1 a scalar.
			self.score = T.sum(T.dot(self.output, scorer.T))
			randScores = [T.sum(T.dot(x, scorer.T)) for x in fake_outputs]

			self.cost = T.sum([T.maximum(0, 1 - self.score + randScore) for randScore in randScores])
			self.costInputs = self.scoreInputs + fake_scoreInputs

	def get_hidden_values(self, value_in, W, b, activation='sigmoid'):
		if activation == 'sigmoid':
			return T.nnet.sigmoid(T.dot(value_in, W) + b)
		elif activation == 'tanh':
			return T.tanh(T.dot(value_in, W) + b)
		else:
			raise NotImplementedError, "%s activation not implemented yet."%activation
	
	def get_output_function(self):
		return theano.function([self.input], self.output)
	
	def get_score_function(self):
		return theano.function(self.scoreInputs, self.score)

	def get_cost_function(self):
		return theano.function(self.costInputs, self.cost)

	def get_output_score_function(self):
		return theano.function([self.output], self.score)

	def get_update_function(self):
		gparams = T.grad(self.cost, self.params)
		self.updates = []
		# NA: [4:] may include leftRep and rightRep for argNodes, and nothing for eventNodes.
		# NA: We want to calculate the gradients with respect to leftRep and rightRep for backprop, but not make any updates.
		for param, gparam in zip(self.params, gparams):
			self.updates.append((param, param - self.lr * gparam))
		return theano.function(self.costInputs, self.cost, updates=self.updates)

class NEM(object):
	def __init__(self, sharedvectors, numargs=5, n_ins=50, n_outs=50, lr=0.001, activation="sigmoid", with_arg_parses=False):
		if with_arg_parses:
			raise NotImplementedError, "Support for pre-defined parses coming soon!"
		numpy_rng = numpy.random.RandomState(89677)
		theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
		self.ts_vocab = theano.shared(value=sharedvectors, name='vocab')
		arg_n_ins = n_ins*2
		arg_n_outs = n_outs
		event_n_ins = n_ins*numargs
		event_n_outs = n_outs
		initial_argbhid = numpy.zeros(arg_n_outs, dtype=theano.config.floatX)
		#initial_argbhid = numpy.zeros(arg_n_ins, dtype=theano.config.floatX)
		#initial_eventbhid = numpy.zeros(event_n_outs, dtype=theano.config.floatX)
		initial_eventbhid = numpy.zeros(event_n_outs, dtype=theano.config.floatX)
		initial_argW = numpy.asarray(numpy_rng.uniform(
					low=-4 * numpy.sqrt(6. / (arg_n_outs + arg_n_ins)),
					high=4 * numpy.sqrt(6. / (arg_n_outs + arg_n_ins)),
					size=(arg_n_ins, arg_n_outs)), dtype=theano.config.floatX)
		initial_eventW = numpy.asarray(numpy_rng.uniform(
					low=-4 * numpy.sqrt(6. / (event_n_outs + event_n_ins)),
					high=4 * numpy.sqrt(6. / (event_n_outs + event_n_ins)),
					size=(event_n_ins, event_n_outs)), dtype=theano.config.floatX)
		initial_argScorer = numpy.asarray(numpy_rng.uniform(
					low=-4 * numpy.sqrt(6. / arg_n_outs),
					high=4 * numpy.sqrt(6. / arg_n_outs),
					size=(1, arg_n_outs)), dtype=theano.config.floatX)
		initial_eventScorer = numpy.asarray(numpy_rng.uniform(
					low=-4 * numpy.sqrt(6. / event_n_outs),
					high=4 * numpy.sqrt(6. / event_n_outs),
					size=(1, event_n_outs)), dtype=theano.config.floatX)
		self.argW = theano.shared(value=initial_argW, name='argW', borrow=True)
		self.argbhid = theano.shared(value=initial_argbhid,
				name='argbhid',
				borrow=True)
		self.eventW = theano.shared(value=initial_eventW, name='eventW', borrow=True)
		self.eventbhid = theano.shared(value=initial_eventbhid,
				name='eventbhid',
				borrow=True)
		self.argScorer = theano.shared(value=initial_argScorer,
				name='argScorer',
				borrow=True)
		self.eventScorer = theano.shared(value=initial_eventScorer,
				name='eventScorer',
				borrow=True)
		self.argLeafNode = AbstractNode(self.ts_vocab, self.argW, self.argbhid, self.argScorer, "arg", arg_n_outs, 
					arg_n_ins, lr=lr, activation=activation)
		self.argLNode = AbstractNode(self.ts_vocab, self.argW, self.argbhid, self.argScorer, "arg", arg_n_outs, 
					arg_n_ins, reps_av = 'l', lr=lr, activation=activation)
		self.argRNode = AbstractNode(self.ts_vocab, self.argW, self.argbhid, self.argScorer, "arg", arg_n_outs, 
					arg_n_ins, reps_av = 'r', lr=lr, activation=activation)
		self.argLRNode = AbstractNode(self.ts_vocab, self.argW, self.argbhid, self.argScorer, "arg", arg_n_outs, 
					arg_n_ins, reps_av = 'lr', lr=lr, activation=activation)
		self.eventNode = AbstractNode(self.ts_vocab, self.eventW, self.eventbhid, self.eventScorer, "event", 
					event_n_outs, event_n_ins, lr=lr, activation=activation)
		self.arg_output_function = self.argLeafNode.get_output_function()
		self.arg_output_score_function = self.argLeafNode.get_output_score_function()
		self.arg_score_function = self.argLeafNode.get_score_function()
		self.arg_cost_function = self.argLeafNode.get_cost_function()
		self.arg_update_function = self.argLeafNode.get_update_function()
		self.arg_l_score_function = self.argLNode.get_score_function()
		self.arg_l_cost_function = self.argLNode.get_cost_function()
		self.arg_l_update_function = self.argLNode.get_update_function()
		self.arg_r_score_function = self.argRNode.get_score_function()
		self.arg_r_cost_function = self.argRNode.get_cost_function()
		self.arg_r_update_function = self.argRNode.get_update_function()
		self.arg_lr_score_function = self.argLRNode.get_score_function()
		self.arg_lr_cost_function = self.argLRNode.get_cost_function()
		self.arg_lr_update_function = self.argLRNode.get_update_function()
		self.event_output_function = self.eventNode.get_output_function()
		self.event_score_function = self.eventNode.get_score_function()
		self.event_cost_function = self.eventNode.get_cost_function()
		# The event update function is different from arg update function, only because of different params.
		self.event_update_function = self.eventNode.get_update_function()

	def buildArgTree(self, wordindexes, vocabsize=0, train=False):
		def getmaxscoremerge(nodes):
			maxscore = -float("inf")
			mergepos = -1
			for i in range(len(nodes)-1):
				score = self.arg_lr_score_function(nodes[i].outputRep, nodes[i+1].outputRep)
				#cost = self.arg_cost_function(dAinput)
				if score > maxscore:
					mergepos = i
					maxscore = score
			return mergepos, maxscore 
		topnodes = []
		# "Fake" representations:  We want to calculate scores of compositions with exactly one word replaced randomly.  So, for every node
		# in the tree, we have a list of fake representations.  For leaf nodes, the list has exactly one element.  For non leaf nodes, the list
		# has as many elements as the number of words the node spans.
		wordembeds = self.ts_vocab.get_value()
		for index in wordindexes:
			if train:
				fakeIndex = randint(0, vocabsize-1)
				newnode = ConcreteArgNode(wordembeds[index], vocabIndex=index, leftChild=None, rightChild=None, fakeReps=[wordembeds[fakeIndex]], fakeIndex=fakeIndex)
				topnodes.append(newnode)
			else:
				newnode = ConcreteArgNode(wordembeds[index], vocabIndex=index, leftChild=None, rightChild=None)
				topnodes.append(newnode)

		#totalcost = 0.0
		maxscore = 0
		while len(topnodes) != 1:
			mergepos, maxscore = getmaxscoremerge(topnodes)
			if mergepos == -1:
				raise RuntimeError, "Could not merge nodes!"
			leftRep = topnodes[mergepos].outputRep
			rightRep = topnodes[mergepos+1].outputRep
			leftNode = topnodes[mergepos]
			rightNode = topnodes[mergepos+1]
			concat_input = numpy.concatenate([leftRep, rightRep], axis=1)
			output = self.arg_output_function(concat_input)

			if train:
				# Logic for creating fake list of current node:  When creating every non-leaf node, compose the actual embedding of left node 
				# with all fake embeddings of right node, followed by composing the actual embedding of the right node with all the fake embeddings of the left node.
				fake_compositions = []
				for fakel in topnodes[mergepos].fakeReps:
					fake_compositions.append(self.arg_output_function(numpy.concatenate([fakel, rightRep], axis=1)))
				for faker in topnodes[mergepos+1].fakeReps:
					fake_compositions.append(self.arg_output_function(numpy.concatenate([leftRep, faker], axis=1)))
				newnode = ConcreteArgNode(output, None, leftNode, rightNode, fake_compositions)
			else:
				# totalcost += mincost
				newnode = ConcreteArgNode(output, None, leftNode, rightNode)
			topnodes = topnodes[:mergepos] + [newnode] + topnodes[mergepos+2:]
			
		#return topnodes, totalcost
		#fake_scores = []
		#if train:
		#	for fakeRep in topnodes[0].fakeReps:
		#		fake_scores.append(self.arg_output_score_function(fakeRep))
		#return topnodes, maxscore, fake_scores
		#TODO: The maxscore here is just the score of composition of the top node.  Is that what we want?
		return topnodes, maxscore

	def train_arg_composition(self, events, vocabsize, no_updates=False):
		# Phase 1 on entire sentences
		trainscores = []
		traincosts = []
		for event in events:
			arg_costs = []
			argnodes, argscore = self.buildArgTree(event, vocabsize, train=True)
			#index += num
			if argnodes[0].leftChild is None and argnodes[0].rightChild is None:
				# It's a one word argument.  We need not train anything here.
				continue
			process_list = [argnodes[0]]
			# Doing a breadth first traversal on the tree to update at each node: BPTS.
			# TODO:  BP has been implemented (inefficiently?) here using the structure from topnodes..
			# to deal with dynamic structures.  
			while len(process_list) != 0:
				topnode = process_list[0]
				#for fakeRep in topnode.fakeReps:
				# Updating parameters for scores obtained by replacing each of the words randomly.
				if topnode.leftChild.vocabIndex is not None and topnode.rightChild.vocabIndex is not None:
					# We are dealing with a node whose both children are leaf nodes
					if no_updates:
						arg_cost = self.arg_cost_function(topnode.leftChild.vocabIndex, topnode.rightChild.vocabIndex, topnode.leftChild.fakeIndex, topnode.rightChild.fakeIndex)
					else:
						arg_cost = self.arg_update_function(topnode.leftChild.vocabIndex, topnode.rightChild.vocabIndex, topnode.leftChild.fakeIndex, topnode.rightChild.fakeIndex)
					arg_costs.append(arg_cost)
				elif topnode.leftChild.vocabIndex is not None and topnode.rightChild.vocabIndex is None:
					for fakeRep in topnode.rightChild.fakeReps:
						if no_updates:
							arg_cost = self.arg_r_cost_function(topnode.leftChild.vocabIndex, topnode.rightChild.outputRep, topnode.leftChild.fakeIndex, fakeRep)
						else:
							arg_cost = self.arg_r_update_function(topnode.leftChild.vocabIndex, topnode.rightChild.outputRep, topnode.leftChild.fakeIndex, fakeRep)
						arg_costs.append(arg_cost)
				elif topnode.leftChild.vocabIndex is None and topnode.rightChild.vocabIndex is not None:
					for fakeRep in topnode.leftChild.fakeReps:
						if no_updates:
							arg_cost = self.arg_l_cost_function(topnode.leftChild.outputRep, topnode.rightChild.vocabIndex, fakeRep, topnode.rightChild.fakeIndex)
						else:
							arg_cost = self.arg_l_update_function(topnode.leftChild.outputRep, topnode.rightChild.vocabIndex, fakeRep, topnode.rightChild.fakeIndex)
						arg_costs.append(arg_cost)
				else:
					for l_fakeRep in topnode.leftChild.fakeReps:
						for r_fakeRep in topnode.rightChild.fakeReps:
							if no_updates:
								arg_cost = self.arg_lr_cost_function(topnode.leftChild.outputRep, topnode.rightChild.outputRep, l_fakeRep, r_fakeRep)
							else:
								arg_cost = self.arg_lr_update_function(topnode.leftChild.outputRep, topnode.rightChild.outputRep, l_fakeRep, r_fakeRep)
							arg_costs.append(arg_cost)
					
				if topnode.leftChild.vocabIndex is None:
					process_list.append(topnode.leftChild)
				if topnode.rightChild.vocabIndex is None:
					process_list.append(topnode.rightChild)
				process_list = process_list[1:]
			#arg_scores.append(argscore)
			trainscores.append(argscore)
			traincosts.extend(arg_costs)
		return trainscores, traincosts

	def get_arg_embeddings(self, events, splitsizes, args_wanted, argindex={}, indexrep = []):
		# arg -> index
		#argindex = {}
		# argindex -> rep
		#indexrep = []
		eventargs = []
		argpositions = {'V':set(), 'A0':set(), 'A1': set(), 'AM-LOC': set(), 'AM-TMP': set()}
		for event, numwords in zip(events, splitsizes):
			if numwords[0] != 1:
				raise ValueError, "Multi word verbs are not handled currently.  The first value in each element of splitsizes should be 1."
			verbindex = event[0]
			argpositions['V'].add(verbindex)
			index=1 # starting at 1 because verb's is the first index
			labelindex = 0
			argnums = []
			# Do away with eventtext, use info in event instead
			for num in numwords[1:]:
				labelindex+=1
				label = args_wanted[labelindex]
				arg_inds = event[index:index+num]
				# Defining a identifier for arg, to help figuring out if a representation for it has been computed already.
				arg_id = " ".join([str(x) for x in arg_inds])
				if arg_id not in argindex:
					argnodes, argscore = self.buildArgTree(arg_inds)
					argrep = argnodes[0].outputRep
					argnum = len(indexrep)
					argindex[arg_id] = argnum
					indexrep.append(argrep)
				else:
					argnum = argindex[arg_id]
					argrep = indexrep[argnum]
				index+=num
				#eventreps.append(argnodes[0].outputRep)
				# Collect the indexes of the args forming the event
				argnums.append(argnum)
				# Collect label specific indexes of argreps for contrastive estimation in phase 2
				argpositions[label].add(argnum)
			eventargs.append((verbindex, argnums))
		#argindex is a dict of argtext -> index ints, indexrep is list of arg representations, eventargs is list of arg indexes of events, argpositions set of argument indexes per label.
		return indexrep, eventargs, argpositions

	def score_event_composition(self, eventargs, indexrep):
		scores = []
		event_embeddings = []
		for verbindex, event in eventargs:
			a0rep, a1rep, a2rep, a3rep = [indexrep[event[x]] for x in range(4)]
			scores.append(self.event_score_function(verbindex, a0rep, a1rep, a2rep, a3rep))
			event_embeddings.append(self.event_output_function(numpy.concatenate([self.ts_vocab.get_value()[verbindex], a0rep, a1rep, a2rep, a3rep], axis=1)))
		return scores, event_embeddings
	
	def train_event_composition(self, indexrep, eventargs, argpositions, labels, no_updates=False):
		# Phase 2
		scores = []
		costs = []
		for (verbindex, event), label in zip(eventargs, labels):
			a0rep, a1rep, a2rep, a3rep = [indexrep[event[x]] for x in range(4)]
			#for fs in fakeScores:
			#	self.event_update_function(verbindex, a0rep, a1rep, a2rep, a3rep, fs)
			if no_updates:
				cost = self.event_cost_function(verbindex, a0rep, a1rep, a2rep, a3rep, label)
			else:
				cost = self.event_update_function(verbindex, a0rep, a1rep, a2rep, a3rep, label) 
			costs.append(cost)
			scores.append(self.event_score_function(verbindex, a0rep, a1rep, a2rep, a3rep))
			
		return scores, costs
