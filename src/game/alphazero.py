from mcts.ai_mcts import MCTS
import torch
import numpy as np
from tqdm import trange
import random
import torch.nn.functional as F

class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()

        while True:
            neural_state = self.game.change_perspective(state, player)
            action_probs = self.mcts.search(neural_state)

            memory.append((neural_state, action_probs, player))

            action = np.random.choice(self.game.action_size, p=action_probs)

            state = self.game.get_next_state(state, action, player)

            value, is_terminal = self.game.get_value_and_terminated(state, action)

            if is_terminal:
                returnMemory = []
                for hist_neural_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else -value
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neural_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory

            player = self.game.get_opponent(player)

    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args["batch_size"]):
            # Sample the memory
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args["batch_size"])]

            # Unzip the sample
            state, policy_targets, value_targets = zip(*sample)
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1,1)

            state = torch.tensor(state, dtype=torch.float32)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32)
            value_targets = torch.tensor(value_targets, dtype=torch.float32)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def learn(self):

        for iteration in range(self.args["num_iterations"]):
            memory = []

            # Play a define number of iterations to create the initial dataset for training
            self.model.eval()
            for selfPlay_iteration in range(self.args["num_selfPlay_iterations"]):
                memory += self.selfPlay()

            # Train with the saving experience
            self.model.train()
            for epoch in trange(self.args["num_epochs"]):
                self.train(memory)

            # Save optimizer and network weights
            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"model_{iteration}.pt")

            