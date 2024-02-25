import chess
import argparse


def main(args):
    with open(args.input) as f:
        data = f.readlines()

    board = chess.Board()
    for line in data:
        move = line.strip()
        board.push_san(move)
    print(board)
    print("board.legal_moves:", board.legal_moves)
    print("board.fen()", board.fen())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default="output/moves.txt")
    parser.add_argument('--output', '-o', default="output/moves.pgn")
    args = parser.parse_args()

    main(args)
