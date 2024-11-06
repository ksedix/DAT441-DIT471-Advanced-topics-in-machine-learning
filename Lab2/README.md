# Tabular Reinforcement Learning assignment

This assignment aims to create your own reinforcement learning agents in tabular environments.
This GitHub repo contains the instructions and files needed for the assignment.
The assignment will use the [gymnasium](https://gymnasium.farama.org/) structure, 
I suggest you familiarize yourselves with this before getting started.

**agent.py** is a template agent for you to fill in.

**riverswim.py** contains the custom environment RiverSwim. '''python3 run_experiment.py --env riverswim:RiverSwim''' will run this environment.

**run_experiment.py** serves as a template for how to run your experiments, it allows you to load different agents and 
environments you might create by running. You can call the existing gymnasium environments or ones you have created yourself.

## General instructions
You will need the gymnasium and numpy libraries. 

You should implement the algorithms yourselves, not using implementations by anybody else. 
****
## Tasks
The tasks consist of describing, implementing, and analyzing Q-learning, Double Q-learning, SARSA and Expected SARSA.
### 1. Describe and compare algorithms
Briefly describe the Q-learning, Double Q-learning, SARSA and Expected SARSA algorithm and how each of them works. Also, briefly describe the dissimilarities and similarities between these agents.

### 2. Initialization of Q-values
Implement Q-learning, Double Q-learning, SARSA and Expected SARSA agents with 5% epsilon greedy policy (95% greedy action) and $\gamma = 0.95$.

Investigate how the initialization of the Q-values affects how well the agents learn to solve RiverSwim (included locally in this repository) and [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/). For instance, initializing Q-tables optimistically (larger than you expect the true table to be, not with zeroes). How does it affect the rewards during training and the greedy policy obtained from the Q-values? How does it affect the exploration? Clearly explain and display evidence supporting your arguments. 

### 3. Run and plot rewards
For every agent, run experiments on both RiverSwim and [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/). Use the best Q-value initialization found in **Task 2**.
Plot the moving average of the rewards while your agent learns, either by episode or step depending on the environment, averaged over 5 runs (i.e., restarting the training 5 times). 
Include error-bars (or something similar) indicating the 95% confidence intervals calculated from your variance of the 5 runs.

### 4. Visualize Q-values and greedy policy
Implement and run (until convergence) value iteration on RiverSwim and [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/). Visualize the Q-values for each state. Draw/visualize the greedy policy obtained from the Q-values. Does it reach the optimal policy?


For every combination of agent and environment in **Task 3**, visualize the Q-values for each state (make sure it is easy to interpret, i.e., not just a table). Draw/visualize the greedy policy obtained from the Q-values. Does it reach the optimal policy? Compare these Q-values and policies with the ones you obtained by value iteration.

*Tip: For [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/), the transition probabilities are available by attribute `env.env.P`.*


### 5. Discussion
What are your conclusions? How do the agents perform? Does the agents perform differently? Discuss and explain if the behaviors are expected based on what you have learned in the course. 

## Grading
The grading will be based on:
1. **Your report**, how well you describe, analyse, and visualize your results and how clear your text is.
2. **Correctness**, how well your solutions perform as expected.

## Tips
- Note that some environments have a **done** flag to indicate that the episode is finished. Make sure you take this into account. 
   - For example, you do not want any future rewards from the terminal state and no transition from the terminal state to the starting state. 
- The explorative policy might look bad due to the $\epsilon$-exploration, good to check if the Q-table seems to give a good policy. You can always compare with the Bellman equation.
- Since FrozenLake is slippery, it might look like a good policy is sub-optimal. Some subtle dynamics are coming from that the agent can only slip perpendicularly to the direction of the action that can be worth thinking about.
- The number of steps in the run_experiments is just an example, you will probably need more steps.
- You will need to have learning rate $\alpha$ *sufficiently small* otherwise there will be high variance ($\alpha=1$ would mean that you ignore the old value and overwrite it).

## Submission
Make sure to include both your code and your report when you submit your assignment. Your report should be submitted as a PDF. PDF and code should be submitted separately, i.e., not combined in a compressed folder. On the other hand, all code can be submitted in a compressed folder. 

