"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


class Error(Exception):
    pass


class InvalidActionForBoard(Error):
    pass


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # checks if endgame, checks if board is empty
    emptyCells = 0
    numElements = 0
    if terminal(board) != True:
        isEmpty = False
        for row in board:
            for element in row:
                if element is EMPTY:
                    emptyCells += 1
                else:
                    numElements += 1

        if emptyCells == 9:
            return "X"
        # if not, checks how many elements are in board and returns the players turn
        else:
            # if the numElements is even, means that is turn for X to move, if not, for O
            if numElements % 2 == 0:
                return "X"
            else:
                return "O"
    else:
        return None


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    # set of possible actions
    actionset = []
    # if the board is completed, or someone won, return none
    if not terminal(board):
        # 0...3
        for i in range(len(board)):
            # 0...3
            for j in range(len(board[0])):
                if board[i][j] is EMPTY:
                    actionset.append(tuple((i, j)))
        return set(actionset)
    else:
        return None


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    try:
        if board[action[0]][action[1]] is not None:
            raise InvalidActionForBoard
        else:
            new_board = copy.deepcopy(board)
            new_board[action[0]][action[1]] = player(board)
            return new_board

    except InvalidActionForBoard:
        print("The action is invalid")


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    threeInARow = 0
    winnerOfGame = ""
    i = 0
    j = 0
    # right diagonal
    if board[i][j] == board[i + 1][j + 1] and board[i][j] == board[i + 2][j + 2] and board[i][j] is not None:
        threeInARow += 1
        winnerOfGame = board[i][j]
    # first row
    if board[i][j] == board[i][j + 1] and board[i][j] == board[i][j + 2] and board[i][j] is not None:
        threeInARow += 1
        winnerOfGame = board[i][j]
    # second row
    if board[i + 1][j] == board[i + 1][j + 1] and board[i + 1][j] == board[i + 1][j + 2] and board[i + 1][
        j] is not None:
        threeInARow += 1
        winnerOfGame = board[i + 1][j]
    # third row
    if board[i + 2][j] == board[i + 2][j + 1] and board[i + 2][j] == board[i + 2][j + 2] and board[i + 2][
        j] is not None:
        threeInARow += 1
        winnerOfGame = board[i + 2][j]
    # first column
    if board[i][j] == board[i + 1][j] and board[i][j] == board[i + 2][j] and board[i][j] is not None:
        threeInARow += 1
        winnerOfGame = board[i][j]
    # second column
    if board[i][j + 1] == board[i + 1][j + 1] and board[i][j + 1] == board[i + 2][j + 1] and board[i][
        j + 1] is not None:
        threeInARow += 1
        winnerOfGame = board[i][j + 1]
    # third column
    if board[i][j + 2] == board[i + 1][j + 2] and board[i][j + 2] == board[i + 2][j + 2] and board[i][
        j + 2] is not None:
        threeInARow += 1
        winnerOfGame = board[i][j + 2]
    # left diagonal
    if board[i][j + 2] == board[i + 1][j + 1] and board[i][j + 2] == board[i + 2][j] and board[i][j + 2] is not None:
        threeInARow += 1
        winnerOfGame = board[i][j + 2]

    if threeInARow == 0 or threeInARow > 1:
        return None
    else:
        return winnerOfGame


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    for i in board:
        for j in i:
            if j is not None:
                continue
            else:
                if winner(board) is None:
                    return False
                else:
                    return True

    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if terminal(board):
        if winner(board) == "O":
            return -1
        elif winner(board) == "X":
            return 1
        else:
            return 0



def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    print(player(board))
    if terminal(board):
        return None
    else:
        if player(board) == "X":
            v = -math.inf
            bestMove = None
            for action in actions(board):
                move = alphaBetaPruning(result(board, action), -math.inf, math.inf, False)
                move = max(move, v)
                if move == 1:
                    return action
                else:
                    if v < move:
                        v = move
                        bestMove = action

            return bestMove
        else:
            v = math.inf
            bestMove = None
            for action in actions(board):
                move = alphaBetaPruning(result(board, action), -math.inf, math.inf, True)
                move = min(move, v)
                if move == -1:
                    return action
                else:
                    if v > move:
                        v = move
                        bestMove = action
            return bestMove



def alphaBetaPruning(state, alpha, beta, maximizingPlayer):
    # if game over or cells are completed,
    if terminal(state):
        return utility(state)

    if maximizingPlayer:
        v = -math.inf
        for child in actions(state):
            eval = alphaBetaPruning(result(state, child), alpha, beta, False)
            v = max(v, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return v
    else:
        v = math.inf
        for child in actions(state):
            eval = alphaBetaPruning(result(state, child), alpha, beta, True)
            v = min(v, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return v
