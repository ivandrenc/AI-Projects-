import itertools
import random
import copy


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        # IF THE NUMBER OF CELLS ARE EQUAL TO THE COUNT OF MINES, THEN ALL ARE MINES
        if len(self.cells) == self.count:
            return self.cells

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if self.count == 0:
            return self.cells

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """

        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            self.cells.remove(cell)


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        self.moves_made.add(cell)

        # coordinates of the cell
        i = cell[0]
        j = cell[1]
        # neighbor cells stored in list
        neighbors = list()
        # first neighbor cell
        neighbor_i = i - 1
        neighbor_j = j - 1

        # Add the neighboring cells
        # checks 2 rows and 1 diagonal
        for k in range(3):
            # initializes each time a row is changed
            neighbor_j = j - 1
            # checks each cell of the row
            for n in range(3):
                # check if the cell is out of the edges
                if neighbor_i > self.height - 1 or neighbor_i < 0 or neighbor_j > self.width - 1 or neighbor_j < 0:
                    neighbor_j = neighbor_j + 1

                # if the cell is in the board, check if its in made moves
                else:
                    if (neighbor_i, neighbor_j) in self.moves_made:
                        neighbor_j = neighbor_j + 1
                        print("cell in moves made")
                    # if the cell is not in made moves, then append it to neighbor cells list
                    else:
                        neighbors.append(((neighbor_i, neighbor_j)))
                        neighbor_j = neighbor_j + 1
            neighbor_i = neighbor_i + 1

        # Add neighboring cells to knowldege
        print("cells neighbors: ", neighbors)
        self.knowledge.append(Sentence(neighbors, count))
        # mark cell as safe in knowledge
        self.mark_safe(cell)

        # check if some cells can be marked as safe or mines based on the sentences in knowledge
        base = copy.deepcopy(self.knowledge)
        # iterates through the sentences in knowledge and checks if all cells are manes or safes
        for sentence in self.knowledge:
            if len(sentence.cells) == sentence.count:
                cells = copy.deepcopy(sentence.cells)
                for cell in cells:
                    self.mark_mine(cell)
                del sentence
            else:
                if sentence.count == 0:
                    cells = copy.deepcopy(sentence.cells)
                    for cell in cells:
                        self.mark_safe(cell)
                    del sentence
        # checks for inferences based on the subsets explained in the background
        for sentence in base:
            if sentence.count == 0:
                continue
            for sentence2 in base:
                if sentence == sentence2 or sentence2.count == 0:
                    continue
                subset = sentence.cells
                subset2 = sentence2.cells
                subsetCount = sentence.count
                subset2Count = sentence2.count

                if subset.issubset(subset2):
                    newSet = subset2.difference(subset)
                    newCount = subset2Count - subsetCount
                    for move in newSet:
                        if move in self.moves_made:
                            newSet.remove(move)
                    if newSet:
                        self.knowledge.append(Sentence(list(newSet), newCount))

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        if self.safes:
            for cell in self.safes:
                if cell not in self.moves_made and cell not in self.mines:
                    return cell
        else:
            return None

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        for k in range(3):
            i = random.randrange(self.height)
            j = random.randrange(self.width)
            if not (i, j) in self.moves_made and not (i, j) in self.safes and not (i, j) in self.mines:
                move = (i, j)
                return move
        return None
