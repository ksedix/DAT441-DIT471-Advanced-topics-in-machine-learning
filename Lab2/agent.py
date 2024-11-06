import numpy as np

class Agent(object):
        """The world's simplest agent!"""
        def __init__(self, state_space, action_space, algorithm = "Q-Learning", epsilon=0.05, gamma=0.95, alpha = 0.1):
            self.action_space = action_space
            self.state_space = state_space
            self.algorithm = algorithm #this is the algorithm used, on-policy or off-policy control algorithm
            self.epsilon = epsilon #this is the exploration probability
            self.epsilon_decay = 0.999
            self.epsilon_min = 0.01
            self.gamma = gamma #this is the discount rate
            self.alpha = 0.1 # this is the learning rate
            #self.Q_values = np.zeros((self.state_space,self.action_space)) #each state has some actions associated with it
            #self.Q_values = np.random.rand(self.state_space, self.action_space) * 5
            #optimistically. This is important for RiverSwim!
            self.Q_values = np.ones((self.state_space,self.action_space)) * 5
            if (self.algorithm == "Double-Q-Learning"):
                #For FrozenLake
                self.Q1_values = np.zeros((self.state_space,self.action_space))
                self.Q2_values = np.zeros((self.state_space,self.action_space))
                #For RiverSwim
                #self.Q1_values = np.ones((self.state_space,self.action_space)) * 5
                #self.Q2_values = np.ones((self.state_space,self.action_space)) * 5



        def observe(self, observation, reward, done):
            #Add your code here
            if (self.algorithm == "Q-Learning"):
                self.Q_values[self.previous_state,self.previous_action] += self.alpha * \
                (reward + self.gamma*(1-done)*np.max(self.Q_values[observation])-self.Q_values[self.previous_state, self.previous_action])
                #print(f"Q-values for state {observation}: {self.Q_values[observation]}")
            elif (self.algorithm == "Double-Q-Learning"):
                coin_flip = np.random.randint(0, 2)
                if (coin_flip == 0):
                    self.Q1_values[self.previous_state, self.previous_action] += self.alpha * \
                    (reward + self.gamma*(1-done)*self.Q2_values[observation,np.argmax(self.Q1_values[observation])] - \
                    self.Q1_values[self.previous_state, self.previous_action])
                else:
                    self.Q2_values[self.previous_state, self.previous_action] += self.alpha * \
                    (reward + self.gamma*(1-done)*self.Q1_values[observation,np.argmax(self.Q2_values[observation])] - \
                    self.Q2_values[self.previous_state, self.previous_action])
            elif (self.algorithm == "SARSA"):
                # Assuming self.Q_values[observation] is the Q-values for the current state
                q_values = self.Q_values[observation]
                # Step 1: Find the maximum Q-value
                max_value = np.max(q_values)
                # Step 2: Find all indices where the Q-value is equal to the maximum
                max_indices = np.flatnonzero(q_values == max_value)
                # Step 3: Randomly select one index from the max_indices
                greedy_action = np.random.choice(max_indices)
                # Generate a random number to decide if we explore or exploit
                if np.random.rand() < self.epsilon:  # Epsilon probability
                    # Create a list of non-greedy actions
                    non_greedy_actions = [a for a in range(self.action_space) if a != greedy_action]
                    # Choose randomly from non-greedy actions
                    action = np.random.choice(non_greedy_actions)
                else:
                # Choose the greedy action (exploitation)
                    action = greedy_action

                self.Q_values[self.previous_state, self.previous_action] += self.alpha * \
                (reward + self.gamma*(1-done)*(self.Q_values[observation,action])-self.Q_values[self.previous_state, self.previous_action])
                #EXPECTED SARSA
            elif self.algorithm=="Expected SARSA":
                # Calculate expected Q-value for the next state
                greedy_action = np.argmax(self.Q_values[observation])
                N = self.action_space  # Total number of actions
                # Calculate expected Q-value
                expected_Q_value = (1 - self.epsilon) * self.Q_values[observation,greedy_action]
                # Add contributions from non-greedy actions
                for action in range(N):
                    if action != greedy_action:
                        expected_Q_value += (self.epsilon / (N-1)) * self.Q_values[observation,action]

                self.Q_values[self.previous_state, self.previous_action] += self.alpha * \
                (reward + self.gamma*(1-done)*expected_Q_value-self.Q_values[self.previous_state, self.previous_action])
            else:
                raise Exception("Invalid algorithm")

        def act(self, observation):
            # Add your code here
            self.previous_state = observation
            if self.algorithm == "Double-Q-Learning":
                self.Q_values = np.add(self.Q1_values,self.Q2_values)
            # Find the greedy action
            # Assuming self.Q_values[observation] is the Q-values for the current state
            q_values = self.Q_values[observation]
            # Step 1: Find the maximum Q-value
            max_value = np.max(q_values)
            # Step 2: Find all indices where the Q-value is equal to the maximum
            max_indices = np.flatnonzero(q_values == max_value)
            # Step 3: Randomly select one index from the max_indices
            greedy_action = np.random.choice(max_indices)
            # Generate a random number to decide if we explore or exploit
            if np.random.rand() < self.epsilon:  # Epsilon probability
                # Create a list of non-greedy actions
                non_greedy_actions = [a for a in range(self.action_space) if a != greedy_action]
                # Choose randomly from non-greedy actions
                action = np.random.choice(non_greedy_actions)
            else:
                # Choose the greedy action (exploitation)
                action = greedy_action
            #self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            self.previous_action = action
            return action