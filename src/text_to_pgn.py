import chess
import chess.pgn
import argparse
import Levenshtein
import pandas as pd


def calculate_similarity(text1, text2):
    distance = Levenshtein.distance(text1, text2)
    max_length = max(len(text1), len(text2))
    normalized_similarity = 1 - (distance / max_length)
    return normalized_similarity


def main(args):
    df = pd.read_csv(args.input)

    board = chess.Board()
    for idx in df.index:
        path = df['img_path'][idx]
        unconf = df['prediction_unconfidence'][idx]
        pred_text = df['predicted_text'][idx]
        print(path, unconf, pred_text)

        move = pred_text.strip()

        # Find most similar legal move to the pred_text
        legal_moves = board.legal_moves
        legal_moves = [board.san(move) for move in list(board.legal_moves)]
        print(legal_moves)
        max_similar_move = legal_moves[0]
        max_similar_similarity = 0
        for legal_move in legal_moves:
            similarity = calculate_similarity(pred_text, legal_move)
            if similarity > max_similar_similarity:
                max_similar_similarity = similarity
                max_similar_move = legal_move
            print("legal_move, similarity",
                  legal_move, similarity)
        print(board)
        print("max_similar_move, max_similar_similarity",
              max_similar_move, max_similar_similarity)

        board.push_san(max_similar_move)

    # Setup PGN object to output
    game = chess.pgn.Game().from_board(board)
    game.headers["Event"] = "Text Event"
    # TODO: set following headers
    # # [Site "?"]
    # # [Date "????.??.??"]
    # # [Round "?"]
    # # [White "?"]
    # # [Black "?"]
    # # [Result "*"]

    # Output pgn file
    with open(args.output, 'w') as f:
        print(game, file=f, end="\n\n")
    print('game:', game)


def calculate_normalized_similarity(text1, text2):
    distance = Levenshtein.distance(text1, text2)
    max_length = max(len(text1), len(text2))
    normalized_similarity = 1 - (distance / max_length)
    return normalized_similarity


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default="output/001_0_pred_sorted_v2.txt")
    # TODO: chess record should exist 2 set, for each white and black player
    #       Add --input_2 and compare the unconf to use the more reliable move
    parser.add_argument('--output', '-o', default="output/001_0_pred.pgn")
    args = parser.parse_args()

    main(args)
