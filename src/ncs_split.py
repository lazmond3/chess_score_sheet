import argparse
from pprint import pprint
import cv2
import pdb
import os
import matplotlib.pyplot as plt
import time


NOTATION_AREA_UPPER = 260
NOTATION_AREA_LOWER = 1800
NOTATION_AREA_LEFT = 150
NOTATION_AREA_RIGHT = 1280

MOVE_ID_WIDTH = 50


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
        game_id, sheet_id, move_id, color = line.split('_')
        color = color.split(" ")[0].strip(".png")
        move = line.split(' ')[-1]
        if game_id not in ret.keys():
            ret[game_id] = {}
        if sheet_id not in ret[game_id].keys():
            ret[game_id][sheet_id] = {}
        move_name = move_id + "_" + color
        ret[game_id][sheet_id][move_name] = move

    return ret


def split_sheet_image_into_move(sheet_image_path):
    # load data
    sheet_image = cv2.imread(sheet_image_path, cv2.IMREAD_GRAYSCALE)
    if sheet_image is None:
        return []

    move_rect_list = []  # (x1, y1, x2, y2)
    img_height, img_width = sheet_image.shape

    sheet_image = sheet_image[NOTATION_AREA_UPPER:NOTATION_AREA_LOWER,
                              NOTATION_AREA_LEFT:NOTATION_AREA_RIGHT]
    img_height, img_width = sheet_image.shape

    # split sheet into move
    move_height, move_width = (img_height // 30), (((img_width // 2) - MOVE_ID_WIDTH) // 2)
    move_height_margin = int(move_height * 0.1)
    # TODO: refactor this block
    # NOTE: why y margin bigger? <- Player used to write move near lower line
    move_num_on_sheet = 60
    for move_idx in range(move_num_on_sheet):
        if move_idx < 30:
            # white move
            x1, y1 = 0, move_height * move_idx
            x2, y2 = x1 + move_width, y1 + move_height
            y1 = max(0, y1 - move_height_margin)
            y2 = min(img_height, y2 + move_height_margin * 2)
            x1, x2 = x1 + MOVE_ID_WIDTH, x2 + MOVE_ID_WIDTH
            move_rect_list.append((x1, y1, x2, y2))

            # black move
            x1, y1 = move_width, move_height * move_idx
            x2, y2 = x1 + move_width, y1 + move_height
            y1 = max(0, y1 - move_height_margin)
            y2 = min(img_height, y2 + move_height_margin * 2)
            x1, x2 = x1 + MOVE_ID_WIDTH, x2 + MOVE_ID_WIDTH
            move_rect_list.append((x1, y1, x2, y2))
        else:
            # white move
            x1, y1 = (img_width // 2), move_height * (move_idx - 30)
            x2, y2 = x1 + move_width, y1 + move_height
            y1 = max(0, y1 - move_height_margin)
            y2 = min(img_height, y2 + move_height_margin * 2)
            x1, x2 = x1 + MOVE_ID_WIDTH, x2 + MOVE_ID_WIDTH
            move_rect_list.append((x1, y1, x2, y2))

            # black move
            x1, y1 =  (img_width // 2) + move_width, move_height * (move_idx - 30)
            x2, y2 = x1 + move_width, y1 + move_height
            y1 = max(0, y1 - move_height_margin)
            y2 = min(img_height, y2 + move_height_margin * 2)
            x1, x2 = x1 + MOVE_ID_WIDTH, x2 + MOVE_ID_WIDTH
            move_rect_list.append((x1, y1, x2, y2))

    # Split and output images for each move
    move_image_dir = sheet_image_path + '_moves/'
    os.makedirs(move_image_dir, exist_ok=True)
    sheet_image_path = move_image_dir + ''
    move_image_path_list = []
    # FIXME limit upto the max move ID
    for move_idx in range(move_num_on_sheet * 2):
        x1, y1, x2, y2 = move_rect_list[move_idx]
        img_cropped = sheet_image[y1:y2, x1:x2]

        color = "white" if move_idx % 2 == 0 else "black"
        move_name = str((move_idx // 2) + 1) + "_" + color
        move_image_path = move_image_dir + '/' + move_name + '.png'
        move_image_path_list.append(move_image_path)
        cv2.imwrite(move_image_path, img_cropped)

    return move_image_path_list


def main(args):
    image_path_list = os.listdir(args.input)
    for image_path in image_path_list:
        split_sheet_image_into_move(args.input + '/' + image_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default="data/kanagawa_champ_2023/images")
    # parser.add_argument('--output', default="move_dir/")
    args = parser.parse_args()

    main(args)
