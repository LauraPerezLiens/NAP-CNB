#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import json
import re

import pandas as pd
import tensorflow as tf

import modeling
import tokenization


# Desactivar eager execution para compatibilidad con TF1
tf.compat.v1.disable_eager_execution()

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Flags adaptado a TF2 compat.v1
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None, "")
flags.DEFINE_string("output_file", None, "")
flags.DEFINE_string("layers", "-1,-2,-3,-4", "")
flags.DEFINE_string(
    "bert_config_file",
    None,
    "The config json file corresponding to the pre-trained BERT model."
)
flags.DEFINE_integer(
    "max_seq_length",
    128,
    "Maximum input sequence length after tokenization."
)
flags.DEFINE_string(
    "init_checkpoint",
    None,
    "Initial checkpoint (usually from a pre-trained BERT model)."
)
flags.DEFINE_string(
    "vocab_file",
    None,
    "Vocabulary file that the BERT model was trained on."
)
flags.DEFINE_bool(
    "do_lower_case",
    False,
    "Whether to lower case the input text."
)
flags.DEFINE_integer("batch_size", 32, "Batch size for predictions.")
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_string("master", None, "If using a TPU, the address of the master.")
flags.DEFINE_integer("num_tpu_cores", 8, "Total number of TPU cores to use if TPU is True.")
flags.DEFINE_bool(
    "use_one_hot_embeddings",
    False,
    "If True, use tf.one_hot for embedding lookups."
)


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


def input_fn_builder(features, seq_length):
    """Creates an input_fn closure to be passed to Estimator."""

    def input_fn(params):
        batch_size = params["batch_size"]

        def gen():
            for feature in features:
                yield {
                    "unique_ids": feature.unique_id,
                    "input_ids": feature.input_ids,
                    "input_mask": feature.input_mask,
                    "input_type_ids": feature.input_type_ids,
                }

        d = tf.data.Dataset.from_generator(
            gen,
            output_types={
                "unique_ids": tf.int32,
                "input_ids": tf.int32,
                "input_mask": tf.int32,
                "input_type_ids": tf.int32,
            },
            output_shapes={
                "unique_ids": (),
                "input_ids": (seq_length,),
                "input_mask": (seq_length,),
                "input_type_ids": (seq_length,),
            },
        )

        d = d.batch(batch_size, drop_remainder=False)
        return d

    return input_fn


def model_fn_builder(bert_config, init_checkpoint, layer_indexes, use_tpu, use_one_hot_embeddings):
    """Returns model_fn closure for Estimator."""

    def model_fn(features, labels, mode, params):
        del labels, params, use_tpu  # no usados

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        input_type_ids = features["input_type_ids"]

        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=input_type_ids,
            use_one_hot_embeddings=use_one_hot_embeddings,
        )

        if mode != tf.estimator.ModeKeys.PREDICT:
            raise ValueError("Only PREDICT mode is supported: %s" % mode)

        tvars = tf.compat.v1.trainable_variables()

        # Carga de checkpoint
        checkpoint = tf.train.load_checkpoint(init_checkpoint)

        assign_ops = []
        for var in tvars:
            name = var.name.split(":")[0]
            if checkpoint.has_tensor(name):
                value = checkpoint.get_tensor(name)
                assign_ops.append(tf.compat.v1.assign(var, value))
                tf.compat.v1.logging.info("Loaded: %s", name)
            else:
                tf.compat.v1.logging.info("Skipped: %s", name)

        with tf.control_dependencies(assign_ops):
            all_layers = model.get_all_encoder_layers()

            predictions = {
                "unique_id": unique_ids,
            }

            for i, layer_index in enumerate(layer_indexes):
                predictions["layer_output_%d" % i] = tf.identity(all_layers[layer_index])

        output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        return output_spec

    return model_fn


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of InputFeatures."""
    features = []

    for ex_index, example in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

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
        input_mask = [1] * len(input_ids)

        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 5:
            tf.compat.v1.logging.info("*** Example ***")
            tf.compat.v1.logging.info("unique_id: %s", example.unique_id)
            tf.compat.v1.logging.info(
                "tokens: %s",
                " ".join([tokenization.printable_text(x) for x in tokens])
            )
            tf.compat.v1.logging.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            tf.compat.v1.logging.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            tf.compat.v1.logging.info(
                "input_type_ids: %s",
                " ".join([str(x) for x in input_type_ids])
            )

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids,
            )
        )

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(input_file):
    """
    Lee un CSV con columnas:
      - epitope_sequence
      - secondary_structure_fragment

    y crea InputExample(text_a=epitope_sequence, text_b=secondary_structure_fragment)
    """
    df = pd.read_csv(input_file)

    required_cols = ["epitope_sequence", "secondary_structure_fragment"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Falta la columna {col} en {input_file}")

    examples = []
    unique_id = 0

    for row in df.itertuples(index=False):
        text_a = str(row.epitope_sequence).strip()
        text_b = str(row.secondary_structure_fragment).strip()

        if not text_a or not text_b:
            continue

        examples.append(
            InputExample(
                unique_id=unique_id,
                text_a=text_a,
                text_b=text_b,
            )
        )
        unique_id += 1

    return examples


def main(_):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    layer_indexes = [int(x) for x in FLAGS.layers.split(",")]

    with open(FLAGS.bert_config_file, "r", encoding="utf-8") as f:
        bert_config_dict = json.load(f)
    bert_config = modeling.BertConfig(**bert_config_dict)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file,
        do_lower_case=FLAGS.do_lower_case
    )

    examples = read_examples(FLAGS.input_file)
    features = convert_examples_to_features(
        examples=examples,
        seq_length=FLAGS.max_seq_length,
        tokenizer=tokenizer
    )

    unique_id_to_feature = {f.unique_id: f for f in features}

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        layer_indexes=layer_indexes,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_one_hot_embeddings,
    )

    session_config = tf.compat.v1.ConfigProto()
    session_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(session_config=session_config)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={"batch_size": FLAGS.batch_size},
    )

    input_fn = input_fn_builder(features=features, seq_length=FLAGS.max_seq_length)

    with tf.io.gfile.GFile(FLAGS.output_file, "w") as writer:
        for result in estimator.predict(input_fn, yield_single_examples=True):
            unique_id = int(result["unique_id"])
            feature = unique_id_to_feature[unique_id]

            for i, token in enumerate(feature.tokens):
                values_txt = None

                for j, layer_index in enumerate(layer_indexes):
                    del layer_index
                    layer_output = result["layer_output_%d" % j]

                    layers = collections.OrderedDict()
                    layers["index"] = layer_indexes[j]
                    layers["values"] = [round(float(x), 6) for x in layer_output[i:(i + 1)].flat]

                    values_txt = [round(float(x), 6) for x in layer_output[i:(i + 1)].flat]

                if token == "[CLS]":
                    writer.write("\t".join(str(item) for item in values_txt))
                    writer.write("\n")


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("init_checkpoint")
    flags.mark_flag_as_required("output_file")
    tf.compat.v1.app.run()