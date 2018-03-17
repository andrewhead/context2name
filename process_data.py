"""
Transforms input data to use word IDs instead of text, and outputs the
transformed data to <output_directory>/<input_basename>_processed.txt.
Also deletes the "0MID" token from each input sequence, as we don't
expect it will carry an inherent data.
"""

import argparse
import json
import os
from tqdm import tqdm


def process_data_file(
        data_file_path, input_token_to_id_map, output_token_to_id_map,
        context_size, sequences_per_example, output_path, show_progress):
    """ Replace text in training data with dictionary indexes. """

    with open(data_file_path) as data_file,\
         open(output_path, 'w') as output_file:

        # Initialize iterator for looping over lines in file
        line_iterator = data_file
        if show_progress:
            line_iterator = tqdm(
                data_file,
                desc="Processing data file, current line")

        for line in line_iterator:
            example = json.loads(line.strip())

            # Replace all input tokens with indexes
            input_sequences = example['input']
            for sequence_index in range(sequences_per_example):

                # If this sequence isn't in the training data, generate on
                # that is nothing but 0PAD tokens.
                if sequence_index >= len(input_sequences):
                    input_sequences.append(
                        (["0PAD"] * context_size) + ["0MID"] +\
                        (["0PAD"] * context_size))
                sequence = input_sequences[sequence_index]

                # Replace tokens with IDs
                for token_index, token in enumerate(sequence):
                    token_id = "UNK"
                    if token in input_token_to_id_map:
                        token_id = input_token_to_id_map[token]
                    sequence[token_index] = token_id

                # Delete the middle token from the list (the placeholder whose
                # name will be predicted), as it does not carry information.
                # This code makes an assumption that the "0MID" token will always
                # appear at the middle of the sequence.
                del sequence[context_size]

            # Replace output tokens with indexes
            example['output'] = output_token_to_id_map[example['output']]

            # Write to the output file (one line at a time)
            output_file.write(json.dumps(example) + "\n")


if __name__ == '__main__':

    # Process program arguments
    PARSER = argparse.ArgumentParser(
        description="Process data to use word IDs instead of token text.")
    PARSER.add_argument(
        'data_file',
        help=(
            "Path to file containing data to process.  Expected to have one " +
            "example per line, in JSON format."),
        )
    PARSER.add_argument(
        '-d',
        '--output-directory',
        type=str,
        help="Path to directory where output should be saved.",
        default="processed",
        )
    PARSER.add_argument(
        '--input-vocabulary-path',
        type=str,
        help="Path to file containing the input vocabulary",
        default=os.path.join("processed", "input_vocabulary.json"),
        )
    PARSER.add_argument(
        '--sequences-per-example',
        type=int,
        help=(
            "The number of sequences that are expected for each example. " +
            "If an example doesn't have this many sequences, it will be " +
            "padded with extra sequences consisting of 0PAD tokens."),
        default=5,
        )
    PARSER.add_argument(
        '--context-size',
        type=int,
        help=(
            "The expected number of tokens on each side of the 0MID token " +
            "in each sequence.  Used for generating 0PAD sequences when " +
            "there aren't enough sequences for an example, and for deleting" +
            "0MID tokens from the input data."),
        default=3,
        )
    PARSER.add_argument(
        '--output-vocabulary-path',
        type=str,
        help="Path to file containing the output vocabulary",
        default=os.path.join("processed", "output_vocabulary.json"),
        )
    PARSER.add_argument(
        '-p',
        '--show-progress',
        action='store_true',
        help="Whether to show progress building the vocabulary.",
        )
    ARGS = PARSER.parse_args()

    OUTPUT_DIRECTORY_PATH = ARGS.output_directory
    if not os.path.exists(OUTPUT_DIRECTORY_PATH):
        os.mkdir(OUTPUT_DIRECTORY_PATH)

    # Load the token-to-ID maps into memory
    with open(ARGS.input_vocabulary_path) as input_vocabulary_file:
        INPUT_TOKEN_TO_ID_MAP = json.load(input_vocabulary_file)
    with open(ARGS.output_vocabulary_path) as output_vocabulary_file:
        OUTPUT_TOKEN_TO_ID_MAP = json.load(output_vocabulary_file)

    # Make the name for an output file from the input file name.
    OUTPUT_PATH = os.path.join(
        OUTPUT_DIRECTORY_PATH,
        (
            os.path.splitext(os.path.basename(ARGS.data_file))[0] +
            "_processed.txt"
        )
    )
    process_data_file(
        ARGS.data_file, INPUT_TOKEN_TO_ID_MAP, OUTPUT_TOKEN_TO_ID_MAP,
        ARGS.context_size, ARGS.sequences_per_example, OUTPUT_PATH,
        ARGS.show_progress)
