# coding=utf-8
# Copyright 2019 The Text2Class Authors.
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Text classifier class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
from datetime import datetime

import tensorflow as tf
import tensorflow_hub as hub

from text2class import optimization
from text2class import tokenization
from text2class import classification

class TextClassifier(object):

    def create_tokenizer_from_hub_module(self):
      """Get the vocab file and casing info from the Hub module."""
      with tf.Graph().as_default():
        bert_module = hub.Module(self.HUB_MODULE_HANDLE)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
          vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                tokenization_info["do_lower_case"]])

      return tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    def __init__(self, 
      num_labels=2, 
      data_column="text",
      label_column="label",
      max_seq_length=128,
      hub_module_handle="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    ):
      """This is a path to an uncased (all lowercase) version of BERT"""
      self.DATA_COLUMN = data_column
      self.LABEL_COLUMN = label_column
      self.NUM_LABELS = num_labels
      self.LABEL_LIST = range(num_labels)
      self.MAX_SEQ_LENGTH = max_seq_length
      self.HUB_MODULE_HANDLE = hub_module_handle
      self.tokenizer = self.create_tokenizer_from_hub_module()


    def create_model(self, is_predicting, input_ids, input_mask, segment_ids, labels, num_labels):
      """Creates a classification model."""

      bert_module = hub.Module(
          self.HUB_MODULE_HANDLE,
          trainable=True)
      bert_inputs = dict(
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids)
      bert_outputs = bert_module(
          inputs=bert_inputs,
          signature="tokens",
          as_dict=True)

      # Use "pooled_output" for classification tasks on an entire sentence.
      # Use "sequence_outputs" for token-level output.
      output_layer = bert_outputs["pooled_output"]

      hidden_size = output_layer.shape[-1].value

      # Create our own layer to tune for politeness data.
      output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

      output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

      with tf.variable_scope("loss"):

        # Dropout helps prevent overfitting
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        # Convert labels into one-hot encoding
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
        # If we're predicting, we want predicted labels and the probabiltiies.
        if is_predicting:
          return (predicted_labels, log_probs)

        # If we're train/eval, compute loss between predicted and actual label
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, predicted_labels, log_probs)


    # model_fn_builder actually creates our model function
    # using the passed parameters for num_labels, learning_rate, etc.
    def model_fn_builder(self, num_labels, learning_rate=None, num_train_steps=None, num_warmup_steps=None):
      """Returns `model_fn` closure for TPUEstimator."""
      def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
        
        # TRAIN and EVAL
        if not is_predicting:

          (loss, predicted_labels, log_probs) = self.create_model(
            is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

          train_op = optimization.create_optimizer(
              loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

          # Calculate evaluation metrics. 
          def metric_fn(label_ids, predicted_labels):
            accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
            return {
                "eval_accuracy": accuracy
            }

          eval_metrics = metric_fn(label_ids, predicted_labels)

          if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode=mode,
              loss=loss,
              train_op=train_op)
          else:
              return tf.estimator.EstimatorSpec(mode=mode,
                loss=loss,
                eval_metric_ops=eval_metrics)
        else:
          (predicted_labels, log_probs) = self.create_model(
            is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

          predictions = {
              'probabilities': log_probs,
              'labels': predicted_labels
          }
          return tf.estimator.EstimatorSpec(mode, predictions=predictions)

      # Return the actual model function in the closure
      return model_fn


    def fit(self, df, 
        BATCH_SIZE=32, # Compute train and warmup steps from batch size
        LEARNING_RATE=2e-5, 
        NUM_TRAIN_EPOCHS=3.0, 
        WARMUP_PROPORTION=0.1, # Warmup is a period of time where the learning rate is small and gradually increases--usually helps training.
        SAVE_CHECKPOINTS_STEPS=500, # number of checkpoint steps between saves
        SAVE_SUMMARY_STEPS=100,
        MODEL_DIR = 'model_out' # # Specify output directory
      ):

      # Use the InputExample class from BERT's classification code to create examples from the data
      train_InputExamples = df.apply(lambda x: classification.InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example
                                                                         text_a = x[self.DATA_COLUMN], 
                                                                         text_b = None, 
                                                                         label = x[self.LABEL_COLUMN]), axis = 1)
      # Convert our train and test features to InputFeatures that BERT understands.
      train_features = classification.convert_examples_to_features(train_InputExamples, self.LABEL_LIST, self.MAX_SEQ_LENGTH, self.tokenizer)
      
      # Compute # train and warmup steps from batch size
      num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
      num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
      
      run_config = tf.estimator.RunConfig(
          model_dir=MODEL_DIR,
          save_summary_steps=SAVE_SUMMARY_STEPS,
          save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

      model_fn = self.model_fn_builder(
        num_labels=self.NUM_LABELS,
        learning_rate=LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps)

      self.estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={"batch_size": BATCH_SIZE})
        
      # Create an input function for training. drop_remainder = True for using TPUs.
      train_input_fn = classification.input_fn_builder(
          features=train_features,
          seq_length=self.MAX_SEQ_LENGTH,
          is_training=True,
          drop_remainder=False)
          
      print(f'Beginning training...')
      current_time = datetime.now()
      self.estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
      print("Finished training in ", datetime.now() - current_time)
      
    def predict(self, in_sentences, BATCH_SIZE=32, MODEL_DIR='model_out'):
      run_config = tf.estimator.RunConfig(model_dir=MODEL_DIR)

      model_fn = self.model_fn_builder(num_labels=self.NUM_LABELS)

      self.estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={"batch_size": BATCH_SIZE})
    
      input_examples = [classification.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in in_sentences] # here, "" is just a dummy label
      input_features = classification.convert_examples_to_features(input_examples, self.LABEL_LIST, self.MAX_SEQ_LENGTH, self.tokenizer)
      predict_input_fn = classification.input_fn_builder(features=input_features, seq_length=self.MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
      predictions = self.estimator.predict(predict_input_fn)
      return [(sentence, prediction['labels']) for sentence, prediction in zip(in_sentences, predictions)]


