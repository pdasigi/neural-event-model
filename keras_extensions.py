'''
Keras is great, but it makes certain assumptions that do not quite work for NLP problems. We override some of those
assumptions here.
'''
from overrides import overrides

from keras import backend as K
from keras.layers import Embedding, TimeDistributed, Flatten


class AnyShapeEmbedding(Embedding):
    '''
    We just want Embedding to work with inputs of any number of dimensions.
    This can be accomplished by simply changing the output shape computation.
    '''
    @overrides
    def compute_output_shape(self, input_shape):
        return input_shape + (self.output_dim,)


class TimeDistributedRNN(TimeDistributed):
    '''
    The TimeDistributed wrapper in Keras works for recurrent layers as well, except that it does not handle masking
    correctly. In case when the wrapper recurrent layer does not return a sequence, no mask is returned. However,
    when we are time distributing it, it is possible that some sequences are entirely padding, for example, when
    one of the slots being encoded is not present in the input at all. We override masking here.
    '''
    @overrides
    def compute_mask(self, x, input_mask=None):
        # pylint: disable=unused-argument
        if input_mask is None:
            return None
        else:
            return K.any(input_mask, axis=-1)


class MaskedFlatten(Flatten):
    '''
    Flatten does not allow masked inputs. This class does.
    '''
    def __init__(self, **kwargs):
        super(MaskedFlatten, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        # Assuming the output will be passed through a dense layer after this.
        if mask is not None:
            inputs = K.switch(K.expand_dims(mask), inputs, K.zeros_like(inputs))
        return super(MaskedFlatten, self).call(inputs)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        else:
            if K.ndim(mask) == 2:
                # This needs special treatment. It means that the input ndim is 3, and output ndim is 2, thus
                # requiring the mask's ndim to be 1.
                return K.any(mask, axis=-1)
            else:
                return K.batch_flatten(mask)
