import gymnasium as gym
from gymnasium import spaces
import numpy as np

# DO NOT MODIFY
class RiverSwim(gym.Env):
    def __init__(self, n=6, small=5/1000, large=1):
        self.n = n
        self.small = small  # payout for 'backwards' action
        self.large = large  # payout at end of chain for 'forwards' action
        self.state = 0  # Start at beginning of the chain
        #self.steps = 0
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(self.n)
        #self.P = self.create_transition_model()

    def create_transition_model(self):
        # Transition model P[state][action] = (probability, next_state, reward, done)
        P = {s: {a: [] for a in range(self.action_space.n)} for s in range(self.n)}

        for state in range(self.n):
            for action in range(self.action_space.n):
                if action == 0:  # Go left
                    if state == 0:
                        P[state][action] = [(1.0, state, self.small, False)]  # Only stay and get small reward
                    else:
                        P[state][action] = [(1.0, state - 1, 0, False)]  # Move left with no reward
                else:  # Go right
                    if state == 0:
                        P[state][action] = [(0.4, state, 0, False), (0.6, state + 1, 0, False)]  # Probabilities of staying or moving right
                    elif state < self.n - 1:  # Normal case
                        P[state][action] = [
                            (0.05, state - 1, 0, False),  # Move left with small probability
                            (0.6, state, 0, False),       # Stay
                            (0.35, state + 1, 0, False)   # Move right
                        ]
                    else:  # Last state, can only stay or go back
                        P[state][action] = [
                            (0.4, state - 1, 0, False),  # Move back
                            (0.6, state, self.large, True)  # Stay and get large reward
                        ]
        return P

    def step(self, action):
        assert self.action_space.contains(action)
        done = False
        reward = 0
        #self.steps += 1
        if action == 0:  # Go left
            if self.state == 0:
                reward = self.small
            else:
                self.state -= 1
        else:
            if self.state == 0:  # 'forwards': go up along the chain
                self.state = np.random.choice([self.state, self.state + 1], p=[0.4, 0.6])
            elif self.state < self.n - 1:  # 'forwards': go up along the chain
                self.state = np.random.choice([self.state-1, self.state, self.state + 1], p=[0.05, 0.6, 0.35])
            else:
                self.state = np.random.choice([self.state-1, self.state], p=[0.4, 0.6])
                if self.state == self.n-1:
                    reward = self.large
                    #done = True
        #if self.steps > 20:
        #    done = True
        return self.state, reward, done, False, {}


    def reset(self, seed=None, options=None):
        self.state = 0
        #self.steps = 0
        return self.state, {}