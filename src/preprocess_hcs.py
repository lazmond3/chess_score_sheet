import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from pprint import pprint


def get_data_sheet(args):
    """
    Load all the data
    Args:
        args: list of image path (1 path = 1 sheet), list of labels (1 label = 1 sheet)
              (label one or multple rows, and 1 row = 1 move:w)
    """
    words_list = []

    training_tags = open(f"{args.input}/training_tags.txt", "r").readlines()

    ret = {}
    # import pdb;pdb.set_trace()
    for line in training_tags:
        line = line.strip()
        game_id, sheet_id, _, _ = line.split('_')
        move = line.split(' ')[-1]
        if game_id not in ret.keys():
            ret[game_id] = {}
        if sheet_id not in ret[game_id].keys():
            ret[game_id][sheet_id] = []
        ret[game_id][sheet_id].append(move)

    # np.random.shuffle(words_list)

    # split_idx = int(train_ratio["train"] * len(words_list))
    # train_samples = words_list[:split_idx]
    # test_samples = words_list[split_idx:]

    # validation_ratio = train_ratio["validation"] / (1 - train_ratio["train"])
    # val_split_idx = int(validation_ratio * len(test_samples))
    # validation_samples = test_samples[:val_split_idx]
    # test_samples = test_samples[val_split_idx:]

    # # Print splitting results
    # print(f"Number of words_list: {len(words_list)}")
    # print(f"Total training samples: {len(train_samples)}")
    # print(f"Total validation samples: {len(validation_samples)}")
    # print(f"Total test samples: {len(test_samples)}")

    # # Create the list of image path and labels (w/ all the info)

    # base_image_path = os.path.join(args.input, "words")

    # def get_image_paths_and_labels(samples):
    #     paths = []
    #     corrected_samples = []
    #     for (i, file_line) in enumerate(samples):
    #         line_split = file_line.strip()
    #         line_split = line_split.split(" ")

    #         # Each line split will have this format for the corresponding image:
    #         # part1/part1-part2/part1-part2-part3.png
    #         image_name = line_split[0]
    #         partI = image_name.split("-")[0]
    #         partII = image_name.split("-")[1]
    #         img_path = os.path.join(
    #             base_image_path, partI, partI + "-" + partII, image_name + ".png"
    #         )
    #         if os.path.getsize(img_path):
    #             paths.append(img_path)
    #             corrected_samples.append(file_line.split("\n")[0])

    #     return paths, corrected_samples

    # train_img_paths, train_labels = get_image_paths_and_labels(train_samples)
    # validation_img_paths, validation_labels = get_image_paths_and_labels(validation_samples)
    # test_img_paths, test_labels = get_image_paths_and_labels(test_samples)

    return ret


def main(args):
    # sheet_image_paths, sheet_labels = get_data_sheet()  # for each score sheet
    ret = get_data_sheet(args)  # for each score sheet
    pprint(ret)
    # move_image_paths, move_labels = split_sheet_to_move()  # for each move


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default="data/HCS")
    parser.add_argument('--output', '-o', default="output_model/")
    parser.add_argument('--random_seed', '-r', type=int, default=None)
    parser.add_argument('--epoch_num', '-e', type=int, default=1,
                        help='Shold be at least 50 for good accuracy')
    parser.add_argument('--plt_show', '-p', action='store_true')
    args = parser.parse_args()

    main(args)
