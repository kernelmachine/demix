import argparse
import numpy as np
import os


def main(args):

    total_lines = 0
    lang_lines_list = []
    with open(args.files_to_line) as fin:
        for line in fin:
            line = line.strip()
            lines, lang_filename = line.split()
            lang = lang_filename.split('.')[args.lang_position_in_filename]
            assert len(lang_filename.split('.')) == 3, f"{lang}"
            single_lang = lang_filename.split('.')[args.lang_position_in_filename + 1]
            lines = int(lines)
            total_lines += lines
            lang_lines_list.append((lang, single_lang, lines))

    probs = []
    temp_scaled_probs = []
    temp_scaled_probs_sum = 0.0
    for lang, single_lang, lines in lang_lines_list:
        prob = float(lines) / total_lines
        temp_scaled_prob = prob ** ( 1/ args.temp)
        probs.append(prob)
        temp_scaled_probs.append(temp_scaled_prob)
        temp_scaled_probs_sum += temp_scaled_prob

    # temp_scaled_prob_renormalized = []
    # lang_sampled_lines_list = []

    fout = open(os.path.join(args.datadir, 'files_to_resampled_lines.tsv'), 'w')
    for index, temp_scaled_prob in enumerate(temp_scaled_probs):
        final_prob = temp_scaled_prob / temp_scaled_probs_sum
        # temp_scaled_prob_renormalized.append(final_prob)
        # lang_sampled_lines_list.append((lang_lines_list[index][0], int(args.total_lines * final_prob)))
        lang_pair = lang_lines_list[index][0]
        single_lang = lang_lines_list[index][1]
        print(lang_lines_list[index][0] + "\t" +
              str(int(args.total_lines * final_prob)) + "\t" +
              os.path.join(args.datadir, ".".join(["train", lang_pair, single_lang])), file=fout)
    fout.close()



"""
Script that output the num samples for SOM training based on the temperature.
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--files-to-line',
        #default='/large_exp/angelafan/flores/namangoyal/cc100_combined/files_to_line_cleaned.tsv'
    )
    parser.add_argument(
        '--lang-position-in-filename',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--datadir',
        #default='/large_exp/angelafan/flores/namangoyal/cc100_combined/'
    )
    parser.add_argument(
        '--temp',
        type=float,
        default=2.0
    )
    parser.add_argument(
        '--total-lines',
        type=int,
        default=500000000,
    )

    args = parser.parse_args()
    main(args)