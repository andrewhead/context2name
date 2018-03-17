"""
Splits a training file into a training and validation file.  Assumes there
is a separate example on each line.
* <output_directory>/training.<ext>
* <output_directory>/validation.<ext>
"""

import argparse
import math
import os
import numpy as np
from tqdm import tqdm


def split_data_file(
        data_file_path, output_directory_path, validation_ratio, show_progress):
    """ Split a data file into a training and validation file. """

    # Count the number of data points
    example_count = 0
    with open(data_file_path) as data_file:
        for line in data_file:
            example_count += 1

    # Compute how many examples should be included in the validation set
    validation_count = math.floor(example_count * validation_ratio)

    # Split the indexes into a training and testing set
    example_indexes = np.arange(0, example_count)
    np.random.shuffle(example_indexes)
    validation_indexes = example_indexes[:validation_count]

    # Initialize paths to training and validation files
    file_ext = os.path.splitext(data_file_path)[1]
    validation_path = os.path.join(
        output_directory_path, 'validation' + file_ext)
    training_path = os.path.join(
        output_directory_path, 'training' + file_ext)

    # Split the data file into training and validation files
    with open(data_file_path) as data_file,\
         open(validation_path, 'w') as validation_file,\
         open(training_path, 'w') as training_file:

        # Initialize iterator for looping over lines in file
        line_iterator = data_file
        if show_progress:
            line_iterator = tqdm(
                data_file,
                desc="Splitting data file, current line")

        for example_index, line in enumerate(line_iterator):
            if example_index in validation_indexes:
                validation_file.write(line)
            else:
                training_file.write(line)


if __name__ == '__main__':

    # Process program arguments
    PARSER = argparse.ArgumentParser(
        description="Split data file into a training and validation file.")
    PARSER.add_argument(
        'data_file',
        help="Path to file containing data to split.  One example per line.",
        )
    PARSER.add_argument(
        '-d',
        '--output-directory',
        type=str,
        help="Path to directory where output should be saved.",
        default="processed",
        )
    PARSER.add_argument(
        '-r',
        '--validation-ratio',
        type=float,
        help="Ratio of data that should be saved for the validation set",
        default=0.2,
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

    split_data_file(
        ARGS.data_file, OUTPUT_DIRECTORY_PATH, ARGS.validation_ratio,
        ARGS.show_progress)
