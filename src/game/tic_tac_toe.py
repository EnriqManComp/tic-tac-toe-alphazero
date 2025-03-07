import numpy as np

class TicTacToe:
    def __init__(self):
        self.row_count = 3 # Possible row in the game
        self.column_count = 3 # Possible column in the game
        # Possible play options
        self.action_size = self.row_count * self.column_count

    def get_initial_state(self):
        """
            Set an initial state in the game
        """
        # All zeros initial state or empty board
        return np.zeros((self.row_count, self.column_count))
    
    def get_next_state(self, state, action, player):
        """
            Set the next state after play
            Args:
                state: Current state of the board
                action: 0-8 Possible actions to take. This can translate to choose the empty space
                        in the board beginning in the top left position in the board and ending in
                        the bottom right position in the board
                player: Value to asign in the board depend of the current player
        """
        # Translate action choice to row and column matrix for the board
        row = action // self.column_count 
        column = action % self.column_count
        # Update the board
        state[row,column] = player
        # Return the updated board
        return state
    
    def get_valid_actions(self, state):
        """
            Get the valid action in the board
        
        """
        return (state.reshape(-1) == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        """
            Check if the action result in a winning move
        """

        if action == None:
            return False
        
        row = action // self.column_count
        column = action % self.column_count

        player = state[row, column]

        # Check winning options
        return (
            np.sum(state[row, :]) == player * self.column_count
            or np.sum(state[:, column]) == player * self.row_count
            or np.sum(np.diag(state)) == player * self.row_count
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count
        )
    
    def get_value_and_terminated(self, state, action):

        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_actions(state)) == 0:
            return 0, True
        return 0, False
        
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    def change_perspective(self, state, player):
        return state * player
