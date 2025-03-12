from game.tic_tac_toe import TicTacToe
#from mcts.mcts import MCTS # No AI
from mcts.ai_mcts import MCTS
from network.alphazero_net import ResNet
import numpy as np
import torch.optim as optim
from game.alphazero import AlphaZero




if __name__ == "__main__":
    # Define game
    game = TicTacToe()
    # Set initial player
    player = 1
    # Set args
    args = {
        "C": 2,
        "num_searches": 60,
        "num_iterations": 3,
        "num_selfPlay_iterations": 10,
        "num_epochs": 4,
        "batch_size": 64
    }

    # Define the Network
    model = ResNet(game, num_resBlocks=4, num_hidden=64)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define MCTS
    #mcts = MCTS(game, args) # No AI
    #mcts = MCTS(game, args, model)

    # Set initial state
    #state = game.get_initial_state()
    
    # Define AlphaZero algorithm
    alphaZero = AlphaZero(model, optimizer, game, args)
    alphaZero.learn()
    """
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
            #print(mcts_probs)
            action = np.argmax(mcts_probs)
            #print("ACTION: ", action)

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
    """