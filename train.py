"""
Train a model to predict variable names from their context.
"""

import argparse
import os
import tensorflow as tf


def _get_feature_names(sequences_per_example, context_size):
    """ Compute feature names given expected input dimensions. """
    feature_names = []
    for sequence_index in range(sequences_per_example):
        for direction in ["left", "right"]:
            for context_index in range(context_size):
                feature_names.append(
                    "seq{sequence_index}_{direction}{context_index}".format(
                        sequence_index=sequence_index,
                        direction=direction,
                        context_index=context_index,
                    ))
    return feature_names


def input_fn(data_file_path, batch_size, feature_names, column_count):

    def _parse_line(line):
        decoded = tf.decode_csv(line, record_defaults=[[-1]] * column_count)
        label = decoded[0]
        features = decoded[1:]
        return dict(zip(feature_names, features)), label

    # Set up data input as a CSV file reader.
    # Don't shuffle here, as there are enough lines that I don't want to shuffle
    # them all in memory.  Assume examples have already been shuffled.
    dataset = (
        tf.data.TextLineDataset([data_file_path])
        .map(_parse_line) 
        .batch(batch_size)
        )
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def train(
        training_file_path, validation_file_path, sequences_per_example,
        context_size, input_vocabulary_size, output_vocabulary_size,
        embedding_dimensions, hidden_units, batch_size, epochs,
        model_directory_path):
    """ Train the model. """

    # Dynamically compute the feature names from the expected input properties.
    feature_names = _get_feature_names(sequences_per_example, context_size)

    # Compute the number of expected columns in the data from input properties.
    # The `+1` is there for the first, label column.
    column_count = (context_size * 2) * sequences_per_example + 1

    # Create a column for each context token.  Then mark them as sharing
    # a set of embedding parameters.  These are the features.
    columns = []
    for feature_name in feature_names:
        columns.append(tf.feature_column.categorical_column_with_identity(
            feature_name, num_buckets=input_vocabulary_size))
    embeddings = tf.feature_column.shared_embedding_columns(
        columns, embedding_dimensions)

    # Use an off-the-shelf classifier
    estimator = tf.estimator.DNNClassifier(
        feature_columns=embeddings,
        n_classes=output_vocabulary_size,
        hidden_units=hidden_units,
        model_dir=model_directory_path,
    )

    # with tf.Session():
    with tf.train.MonitoredTrainingSession():

        for _ in range(epochs):

            # Reset the training and validation data generators
            training_input_fn = lambda: input_fn(
                training_file_path, batch_size, feature_names, column_count)

            # Train the model
            try:
                estimator.train(
                    input_fn=training_input_fn,
                )
            except tf.errors.OutOfRangeError:
                break


if __name__ == '__main__':

    # Process program arguments
    PARSER = argparse.ArgumentParser(
        description="Train a model to predict variable names from context")
    PARSER.add_argument(
        '-t',
        '--training-file',
        type=str,
        help="Path to file containing training data (CSV).",
        default=os.path.join("processed", "training_shuffled.csv"),
        )
    PARSER.add_argument(
        '-v',
        '--validation-file',
        type=str,
        help="Path to file containing validation data (CSV).",
        default=os.path.join("processed", "validation.csv"),
        )
    PARSER.add_argument(
        '-m',
        '--model-directory',
        type=str,
        help="Path to directory where model and checkpoints should be saved.",
        default="models",
        )
    PARSER.add_argument(
        '-s',
        '--sequences-per-example',
        type=int,
        help="The number of sequences that are expected in each example.",
        default=5,
        )
    PARSER.add_argument(
        '-c',
        '--context-size',
        type=int,
        help=(
            "The number of tokens provided on each side of the output token,"
            "in each sequence."),
        default=3,
        )
    PARSER.add_argument(
        '-b',
        '--batch-size',
        type=int,
        help="Training batch size",
        default=32,
        )
    PARSER.add_argument(
        '-i',
        '--input-vocabulary-size',
        type=int,
        help="Number of tokens in the input vocabulary",
        default=4098,  # 4096 + PAD + UNK
        )
    PARSER.add_argument(
        '-o',
        '--output-vocabulary-size',
        type=int,
        help="Number of tokens in the output vocabulary",
        default=60001,  # 60000 + UNK
        )
    PARSER.add_argument(
        '-e',
        '--embedding-dimensions',
        type=int,
        help="Number of dimensions for embedding input tokens",
        default=8,  # Heuristic: 4096 ^ 0.25 (as suggested in TF documentation)
        )
    PARSER.add_argument(
        '-n',
        '--epochs',
        type=int,
        help="Number of epochs (rounds of training on full dataset)",
        default=100,
        )
    PARSER.add_argument(
        '-u',
        '--hidden-units',
        type=int,
        nargs='+',
        help="Number of dense hidden units in the network.",
        default=[512],
        )
    ARGS = PARSER.parse_args()

    MODEL_DIRECTORY_PATH = ARGS.model_directory
    if not os.path.exists(MODEL_DIRECTORY_PATH):
        os.mkdir(MODEL_DIRECTORY_PATH)

    train(
        training_file_path=ARGS.training_file,
        validation_file_path=ARGS.validation_file,
        sequences_per_example=ARGS.sequences_per_example,
        context_size=ARGS.context_size,
        input_vocabulary_size=ARGS.input_vocabulary_size,
        output_vocabulary_size=ARGS.output_vocabulary_size,
        embedding_dimensions=ARGS.embedding_dimensions,
        hidden_units=ARGS.hidden_units,
        batch_size=ARGS.batch_size,
        epochs=ARGS.epochs,
        model_directory_path=MODEL_DIRECTORY_PATH,
    )
