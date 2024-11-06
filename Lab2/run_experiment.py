import argparse
import gymnasium as gym
import importlib.util

from matplotlib import pyplot as plt
import numpy as np

from riverswim import RiverSwim

parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object", default="agent.py")
parser.add_argument("--env", type=str, help="Environment", default="FrozenLake-v1")
parser.add_argument("--render_mode", type=str, help="Render mode for the environment", default="human")  # Add render_mode argument
args = parser.parse_args()

spec = importlib.util.spec_from_file_location('Agent', args.agentfile)
agentfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentfile)

try:
    # render_mode=args.render_mode
    env = gym.make(args.env)
    print("Loaded ", args.env)
except:
    file_name, env_name = args.env.split(":")
    gym.envs.register(
        id=env_name + "-v0",
        entry_point=args.env,
    )
    # render_mode=args.render_mode
    env = gym.make(env_name + "-v0")
    print("Loaded", args.env)

# Load the RiverSwim environment directly
#env = RiverSwim()  # Instantiating the RiverSwim environment

# Prepare to run multiple experiments
n_runs = 5  # Number of runs
n_episodes = 5000  # Total Number of Episodes (used by FrozenLake)
rewards_list = []  # To store rewards for each run
n_time_steps = 5000 # Total number of time steps (used by RiverSwim)
action_dim = env.action_space.n
state_dim = env.observation_space.n
Q_values_sum = None

#Task 2 and 3
for run in range(n_runs):
    rewards = []
    # Instantiate the agent
    #print("Q-values starts with:\n",agent.Q_values)
    agent = agentfile.Agent(state_dim, action_dim, algorithm="Q-Learning")
    observation = env.reset()[0]
    for i in range(n_episodes):
        #For RiverSwim
        """
        action = agent.act(observation)
        observation, reward, done, truncated, info = env.step(action)
        agent.observe(observation, reward, done)
        rewards.append(reward)
        #For FrozenLake
        """
        total_reward = 0
        while True:
            action = agent.act(observation)
            observation, reward, done, truncated, info = env.step(action)
            agent.observe(observation, reward, done)
            total_reward += reward
            if done:
                rewards.append(total_reward)
                observation, info = env.reset()
                break
    rewards_list.append(rewards)
    #print(rewards)
    # Check if Q_values_sum is initialized
    if Q_values_sum is None:
    # Initialize the Q_values_sum with zeros, with the same shape as the first Q_values
        Q_values_sum = np.zeros_like(agent.Q_values)
    #print("Rewards after run",i,"\n:",rewards)
    Q_values_sum += agent.Q_values
    print(agent.Q_values)

env.close()

# After all runs, calculate the average of Q_values
Q_values_avg = Q_values_sum / n_runs
# Print Q-values in a readable format
print("Q-values average:")

q_values_reshaped = Q_values_avg.reshape(4, 4, 4)  # Reshape to (4, 4, 4) for clarity
for row in range(4):
    q_row = []
    for col in range(4):
        state_q_values = q_values_reshaped[row, col]  # Get Q-values for each state
        q_row.append(f"[{state_q_values[0]:.2f}, {state_q_values[1]:.2f}, {state_q_values[2]:.2f}, {state_q_values[3]:.2f}]")
    print(" | ".join(q_row))  # Print Q-values for each state in the row

# Calculate average rewards and error bars
rewards_array = np.array(rewards_list)  # Shape (n_runs, total_steps)
average_rewards = np.mean(rewards_array, axis=0)
std_rewards = np.std(rewards_array, axis=0)
confidence_interval = 1.96 * std_rewards / np.sqrt(n_runs)  # 95% CI

# Calculate moving average and confidence intervals based on moving average
def calculate_moving_average_and_ci(data, window_size, step_size):
    #Calculates moving average and confidence interval.
    moving_avg = []
    ci_upper = []
    ci_lower = []
    episodes = []
    half_window = window_size // 2

    # Start at episode `step_size`, calculate for every multiple of `step_size`
    for i in range(half_window, len(data) - half_window, step_size):
        start = i - half_window  # 50 episodes before
        end = i + half_window    # 50 episodes after
        window_data = data[start:end]
        avg = np.mean(window_data)
        moving_avg.append(avg)
        # Calculate confidence interval
        std_error = np.std(window_data) / np.sqrt(len(window_data))
        ci_upper.append(avg + 1.96 * std_error)
        ci_lower.append(avg - 1.96 * std_error)
        episodes.append(i)  # Store the center episode for plotting
    return episodes, moving_avg, ci_upper, ci_lower

# Apply the moving average calculation
window_size = 100
step_size = window_size // 2

episodes, rewards_moving_avg, ci_upper, ci_lower = calculate_moving_average_and_ci(average_rewards, window_size, step_size)
# Plot the average rewards with error bars
plt.figure(figsize=(10, 6))
plt.plot(episodes, rewards_moving_avg, label='Moving Average of Rewards', color='blue')
plt.fill_between(episodes, ci_lower, ci_upper, color='blue', alpha=0.2, label='95% Confidence Interval')
plt.xlabel('Episode')
#plt.xlabel("Time step")
plt.ylabel('Average Reward')
plt.title('Average Rewards with Moving Average and Confidence Intervals')
plt.legend()
plt.grid()
plt.show()


#Task 4 - Visualize Q-values and greedy policy
"""
# Unwrap the environment to access the transition probabilities and reward structure
env = env.unwrapped


def build_transition_model(n_states, small_reward=5/1000, large_reward=1):
    P = {s: {a: [] for a in range(2)} for s in range(n_states)}
    for state in range(n_states):
        # Action 0: Go left
        if state == 0:
            P[state][0].append((1.0, state, small_reward, False))  # Staying in the same state with small reward
        else:
            P[state][0].append((1.0, state - 1, 0, False))  # Move to the left state with 0 reward
        # Action 1: Go right
        if state == 0:
            P[state][1].append((0.4, state, 0, False))  # Stay in the same state
            P[state][1].append((0.6, state + 1, 0, False))  # Move to the next state
        elif state < n_states - 1:
            P[state][1].append((0.05, state - 1, 0, False))  # Move left
            P[state][1].append((0.6, state, 0, False))  # Stay in the same state
            P[state][1].append((0.35, state + 1, 0, False))  # Move to the next state
        else:  # Last state
            P[state][1].append((0.4, state - 1, 0, False))  # Move left
            P[state][1].append((0.6, state, 0, False))  # Stay in the same state
            P[state][1].append((0.0, state, large_reward, True))  # Move to the last state with large reward
    return P

# Initialize parameters
n_states = 6  # Number of states in RiverSwim
small_reward = 5 / 1000  # Reward for going left
large_reward = 1  # Reward for reaching the end

# Build transition model
transition_model = build_transition_model(n_states, small_reward, large_reward)

# Value Iteration parameters
gamma = 0.95  # Discount factor
delta = 0.0001  # Threshold for convergence

n_states = env.observation_space.n
n_actions = env.action_space.n

# Initialize the value function to zeros
V = np.zeros(n_states)

def value_iteration(env, gamma=0.95, delta=0.0000001):
    \"""
    Perform Value Iteration on the given environment.
    Args:
    - env: The environment (e.g., FrozenLake or RiverSwim)
    - gamma: Discount factor
    - delta: Threshold for convergence

    Returns:
    - V: The value function after convergence
    - policy: The optimal policy after value iteration
    - q_values: The Q-values for each state-action pair
    \"""
    n_states = env.observation_space.n  # Number of states
    n_actions = env.action_space.n  # Number of actions
    V = np.zeros(n_states)  # Value function initialization
    q_values = np.zeros((n_states, n_actions))  # Q-values initialization

    # Iterate until value function converges
    while True:
        delta_val = 0  # Track maximum change for convergence
        for state in range(n_states):
            v = V[state]  # Store the old value

            # Reset Q-values for the current state
            q_values[state] = np.zeros(n_actions)

            # Perform Bellman update: find the max expected value over all actions
            for action in range(n_actions):
                for prob, next_state, reward, done in env.P[state][action]:
                    q_values[state][action] += prob * (reward + gamma * V[next_state])

            V[state] = np.max(q_values[state])  # Update the value function for the current state

            # Update delta_val to track the maximum difference
            delta_val = max(delta_val, abs(v - V[state]))

        # Check for convergence
        if delta_val < delta:
            break

    # Extract policy from the Q-values
    policy = np.argmax(q_values, axis=1)

    return V, policy, q_values  # Return Q-values along with V and policy

# Perform Value Iteration
V, policy, q_values = value_iteration(env, gamma, delta)
"""


# Assuming V and policy have already been calculated from value_iteration

#For FrozenLake
"""
# Optimal Value Function
print("Optimal Value Function:")
value_grid = V.reshape(4, 4)  # Reshape V to match the 4x4 grid
for row in value_grid:
    print(" | ".join(f"{value:.2f}" for value in row))  # Print each row

# Optimal Policy
print("\nOptimal Policy (0=Left, 1=Down, 2=Right, 3=Up):")
policy_grid = policy.reshape(4, 4)  # Reshape policy to match the 4x4 grid
for row in policy_grid:
    print(" | ".join(f"{action}" for action in row))  # Print each row

q_values_reshaped = q_values.reshape(4, 4, 4)  # Reshape to (4, 4, 4) for clarity
#print(q_values_reshaped)
# Print Q-values in a readable format
print("\nQ-values:")
for row in range(4):
    q_row = []
    for col in range(4):
        state_q_values = q_values_reshaped[row, col]  # Get Q-values for each state
        q_row.append(f"[{state_q_values[0]:.2f}, {state_q_values[1]:.2f}, {state_q_values[2]:.2f}, {state_q_values[3]:.2f}]")
    print(" | ".join(q_row))  # Print Q-values for each state in the row

"""

#For RiverSwim
"""
# Assuming V is the value function and policy is the optimal policy obtained from value iteration
print("Optimal Value Function:")
value_list = [f"State {state}: {value:.2f}" for state, value in enumerate(V)]
print(" | ".join(value_list))  # Print value function as a horizontal list

print("\nOptimal Policy (0=Go Left, 1=Go Right):")
policy_list = [f"State {state}: Action {action}" for state, action in enumerate(policy)]
print(" | ".join(policy_list))  # Print policy as a horizontal list

# Print Q-values
print("\nQ-values:")
q_values_list = [f"State {state}: Q(0)={q_values[state][0]:.2f}, Q(1)={q_values[state][1]:.2f}" for state in range(n_states)]
print(" | ".join(q_values_list))  # Print Q-values as a horizontal list
"""


