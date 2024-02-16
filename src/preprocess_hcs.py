import argparse
from pprint import pprint
import cv2
import pdb


def get_data_sheet(args):
    """
    Load all the data
    Args:
        args: list of image path (1 path = 1 sheet), list of labels (1 label = 1 sheet)
              (label one or multple rows, and 1 row = 1 move:w)
    """
    training_tags = open(f"{args.input}/training_tags.txt", "r").readlines()

    ret = {}
    for line in training_tags:
        line = line.strip()
        game_id, sheet_id, _, _ = line.split('_')
        move = line.split(' ')[-1]
        if game_id not in ret.keys():
            ret[game_id] = {}
        if sheet_id not in ret[game_id].keys():
            ret[game_id][sheet_id] = []
        ret[game_id][sheet_id].append(move)

    return ret


def get_data_move(data_sheet, args):
    """
    Load all the data
    Args:
        args: list of image path (1 path = 1 sheet), list of labels (1 label = 1 sheet)
              (labiel one or multple rows, and 1 row = 1 move:w)
    """
    # pdb.set_trace()
    image_path_list = []  # 1 image = 1 move(=part of score sheet image)
    move_list = []

    for game_id in data_sheet.keys():
        for sheet_id in data_sheet[game_id].keys():
            sheet_image_filename = game_id + "_" + sheet_id + ".png"
            sheet_image_path = args.input + "/" + sheet_image_filename
            move_image_path_list = split_sheet_image_into_move(sheet_image_path, len(data_sheet[game_id][sheet_id]))
            for move_idx, move_image_path in enumerate(move_image_path_list):
                image_path_list.append(move_image_path)
                move_list.append(data_sheet[game_id][sheet_id][move_idx])

    return image_path_list, move_list


def split_sheet_image_into_move(sheet_image_path, move_length):
    sheet_image = cv2.imread(sheet_image_path, cv2.IMREAD_GRAYSCALE)
    # TODO: implement

    return [sheet_image_path + '_move' + str(idx) + '.png' for idx in range(move_length)]  # return dummy images


def main(args):
    data_sheet = get_data_sheet(args)  # for each score sheet
    image_path_list, move_list = get_data_move(data_sheet, args)
    print("image_path_list", image_path_list)


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
