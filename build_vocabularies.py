"""
Outputs a dictionary of common input and output for training data.
Creates two file:
* <output_directory>/input_vocabulary.json.
* <output_directory>/output_vocabulary.json.
"""
import argparse
import json
import os


def _to_word_id_dictionary(token_counts, vocabulary_size):
    """
    Converts a dictionary of form { "word": <count> } to a dictionary
    of { "word": <word-id> } by trunacting the vocabulary to only
    `vocabulary_size` words with the highest counts.  IDs are assigned
    from most frequent (1) to least frequent (`vocabulary_size`).
    """
    count_token_pairs = [(c, t) for (t, c) in token_counts.items()]
    sorted_count_token_pairs =\
        sorted(count_token_pairs, key=lambda p: p[0], reverse=True)

    # Construct a mapping from word IDs to their text
    token_to_id_dictionary = {}
    vocabulary = [
        pair[1] for pair in sorted_count_token_pairs[:vocabulary_size]]
    for (index, token) in enumerate(vocabulary, start=1):
        token_to_id_dictionary[token] = index
    return token_to_id_dictionary


def build_input_vocabulary(
        training_file_path, input_vocabulary_size,
        output_directory_path):
    """ Create input vocabulary dictionary and save to a file. """

    # Count up how often each token appears in the inputs
    token_counts = {}
    with open(training_file_path) as training_file:
        for line in training_file:
            example = json.loads(line.strip())
            input_sequences = example['input']
            for input_sequence in input_sequences:
                for token in input_sequence:
                    # Skip the PAD token.  This will always be in the
                    # vocabulary, regardless of how often it occurs.
                    if token == "0PAD":
                        continue
                    if not token in token_counts:
                        token_counts[token] = 0
                    token_counts[token] += 1

    token_to_id_dictionary = _to_word_id_dictionary(
        token_counts, input_vocabulary_size)

    # Add special tokens "UNK" (unknown) and "0PAD"
    token_to_id_dictionary["0PAD"] = len(token_to_id_dictionary)
    token_to_id_dictionary["UNK"] = len(token_to_id_dictionary)

    # Save the dictionary to file
    output_path = os.path.join(output_directory_path, "input_vocabulary.json")
    with open(output_path, 'w') as output_file:
        json.dump(token_to_id_dictionary, output_file, indent=2)


def build_output_vocabulary(
        training_file_path, output_vocabulary_size,
        output_directory_path):
    """ Create output vocabulary dictionary and save to a file. """

    # Count up how often each token appears in the example output
    token_counts = {}
    with open(training_file_path) as training_file:
        for line in training_file:
            example = json.loads(line.strip())
            token = example['output']
            if not token in token_counts:
                token_counts[token] = 0
            token_counts[token] += 1

    token_to_id_dictionary = _to_word_id_dictionary(
        token_counts, output_vocabulary_size)

    # Add special token "UNK" (unknown)
    token_to_id_dictionary["UNK"] = len(token_to_id_dictionary)

    # Save the dictionary to file
    output_path = os.path.join(output_directory_path, "output_vocabulary.json")
    with open(output_path, 'w') as output_file:
        json.dump(token_to_id_dictionary, output_file, indent=2)


if __name__ == '__main__':

    # Process program arguments
    PARSER = argparse.ArgumentParser(
        description="Build vocabularies for tokens in training data.")
    PARSER.add_argument(
        'training_file',
        help=(
            "Path to file containing training data.  Expected to have one " +
            "training example per line, in JSON format."),
        )
    PARSER.add_argument(
        '-d',
        '--output-directory',
        type=str,
        help="Path to directory where output should be saved.",
        default="processed",
        )
    PARSER.add_argument(
        '-i',
        '--input-vocabulary-size',
        type=int,
        help="Size of input vocabulary",
        default=4096,
        )
    PARSER.add_argument(
        '-o',
        '--output-vocabulary-size',
        type=int,
        help="Size of output vocabulary",
        default=60000,
        )
    ARGS = PARSER.parse_args()

    OUTPUT_DIRECTORY_PATH = ARGS.output_directory
    if not os.path.exists(OUTPUT_DIRECTORY_PATH):
        os.mkdir(OUTPUT_DIRECTORY_PATH)

    build_input_vocabulary(
        ARGS.training_file, ARGS.input_vocabulary_size,
        OUTPUT_DIRECTORY_PATH)

    build_output_vocabulary(
        ARGS.training_file, ARGS.output_vocabulary_size,
        OUTPUT_DIRECTORY_PATH)
