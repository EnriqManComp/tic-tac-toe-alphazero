from game.tic_tac_toe import TicTacToe
from mcts.mcts import MCTS
import numpy as np




if __name__ == "__main__":
    # Define game
    game = TicTacToe()
    # Set initial player
    player = 1
    # Set args
    args = {
        "C": 1.41,
        "num_searches": 1000
    }
    # Set initial state
    state = game.get_initial_state()

    # Define MCTS
    mcts = MCTS(game, args)

    while True:
        print(state)

        if player == 1:
            # Get possible actions on the current board
            valid_actions = game.get_valid_actions(state)
            print("valid actions: ", [i for i in range(game.action_size) if valid_actions[i]==1])
            # Get the action to take
            action = int(input(f"{player}:"))

            # Check if action is a valid action
            if valid_actions[action] == 0:
                print("action not valid")
                continue
        else:
            neutral_state = game.change_perspective(state, player)
            mcts_probs = mcts.search(neutral_state)
            action = np.argmax(mcts_probs)

        state = game.get_next_state(state, action, player)

        value, is_terminal = game.get_value_and_terminated(state, action)

        if is_terminal:
            print(state)
            if value == 1:
                print(player, "won")
            else:
                print("draw")
            break

        player = game.get_opponent(player)