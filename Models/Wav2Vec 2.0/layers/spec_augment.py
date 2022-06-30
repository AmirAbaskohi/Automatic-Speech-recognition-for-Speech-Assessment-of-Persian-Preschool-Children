# following code is largly adapted from `here <https://github.com/huggingface/transformers/blob/f2c4ce7e339f4a2f8aaacb392496bc1a5743881f/src/transformers/models/wav2vec2/modeling_tf_wav2vec2.py#L206>__`

import tensorflow as tf

import numpy as np


def tf_multinomial_no_replacement(distribution, num_samples):
    """
    Categorical sampling without replacement is currently not implemented. The gumbel-max trick will do for now - see
    https://github.com/tensorflow/tensorflow/issues/9260 for more info
    """
    # tf.random generators not working on XLA devices
    random_numbers = np.random.uniform(0, 1, distribution.shape)
    random_numbers = tf.constant(random_numbers, dtype=distribution.dtype)
    z = -1 * tf.math.log(random_numbers)
    _, indices = tf.nn.top_k(distribution + z, num_samples)
    return indices


def _scatter_values_on_batch_indices(values, batch_indices, output_shape):
    """
    Scatter function as in PyTorch with indices in format (batch_dim, indixes)

    adapted from `here <https://github.com/huggingface/transformers/blob/2e5dbdf2db4599a6694d0974575a70f9bc3c978e/src/transformers/models/wav2vec2/modeling_tf_wav2vec2.py#L191>`
    """
    indices_shape = batch_indices.shape
    # broadcast batch dim to indices_shape
    broadcasted_batch_dims = tf.reshape(
        tf.broadcast_to(
            tf.expand_dims(tf.range(indices_shape[0]), axis=-1), indices_shape
        ),
        [1, -1],
    )
    # transform batch_indices to pair_indices
    pair_indices = tf.transpose(
        tf.concat([broadcasted_batch_dims, tf.reshape(batch_indices, [1, -1])], 0)
    )
    # scatter values to pair indices
    return tf.scatter_nd(pair_indices, tf.reshape(values, [-1]), output_shape)


def _compute_mask_indices(shape, mask_prob, mask_length, min_masks=2):

    batch_size, seqlen = shape

    if mask_length > seqlen:
        raise ValueError(
            f"`mask_length` ({mask_length}) must be smaller than `seq_length` ({seqlen})."
        )
    num_mask_spans = int(mask_prob * (seqlen / mask_length) + np.random.rand(1))
    num_mask_spans = max(num_mask_spans, min_masks)

    if num_mask_spans * mask_length > seqlen:
        num_mask_spans = seqlen // mask_length

    distribution = tf.ones((batch_size, seqlen - (mask_length - 1)))

    mask_indices = tf_multinomial_no_replacement(distribution, num_mask_spans)

    mask_indices = tf.broadcast_to(
        mask_indices[:, :, None], (batch_size, num_mask_spans, mask_length)
    )
    mask_indices = tf.reshape(mask_indices, (batch_size, num_mask_spans * mask_length))

    offsets = tf.broadcast_to(
        tf.range(mask_length)[None, None, :], (batch_size, num_mask_spans, mask_length)
    )
    offsets = tf.reshape(offsets, (batch_size, num_mask_spans * mask_length))

    mask_indices += offsets

    mask_indices = _scatter_values_on_batch_indices(
        tf.ones_like(mask_indices), mask_indices, (batch_size, seqlen)
    )

    return mask_indices


def apply_spec_augmentation(features, masked_spec_augment, mask_prob, mask_length):
    """
    This method apply spec-augmentation to the `hidden_states`

    Args:
        features (:obj: `tf.Tensor`) of shape (batch_size, seqlen, hidden_size):
            hidden states which we want to mask.
        masked_spec_augment (:obj: `tf.Tensor`) of shape (hidden_states,):
            replace indices to be masked with these values.
        mask_prob (:obj: `float`):
            probability if certain token should be masked, this decides number of tokens to be masked.
        mask_length (:obj: `int`):
            span length of the tokens to be masked.
    Return:
        features (:obj: `tf.Tensor`) of shape (batch_size, seqlen, hidden_size):
            hidden states masked at certain positions which are chosen randomly.
    """
    mask_indices = _compute_mask_indices(
        features.shape[:2], mask_prob, mask_length, min_masks=2
    )

    mask_indices = tf.cast(mask_indices[:, :, None], tf.bool)

    masked_spec_augment = tf.cast(masked_spec_augment, features.dtype)[None, None, :]

    features = tf.where(mask_indices, masked_spec_augment, features)
    return features
