import random

NUM_EPISODES = 1000
MAX_EPISODE_LENGTH = 500

DEFAULT_DISCOUNT = 0.9
EPSILON = 0.05
LEARNINGRATE = 0.1


class QLearner():
    """
    Q-learning agent
    """

    def __init__(self, num_states, num_actions, epsilon=EPSILON):
        self.name = "agent1"
        self.num_actions = num_actions
        self.epsilon = epsilon
        # The Q table is a dictionary where the key is the state and the value is a list
        # where the action corresponds with the action
        self.qtable = {}

    def process_experience(self, state, action, next_state, reward, done):  # You can add more arguments if you want
        """
        Update the Q-value based on the state, action, next state and reward.
        """
        # Get the current reward for the state and action
        greedy_old = self.qtable[state][action]
        # If the next_state is not yet in qtable we need to add it with a reward of 0 for every action
        if next_state not in self.qtable:
            self.qtable[next_state] = [0] * self.num_actions
        # If this action finishes the episode we use this formula to update Q
        if done:
            self.qtable[state][action] = (1 - LEARNINGRATE) * greedy_old + LEARNINGRATE * reward
        else:
            # Get the best action in the next state and update the Qtable
            max_q_old = max([self.qtable[next_state][i] for i in range(4)])
            self.qtable[state][action] = (1 - LEARNINGRATE) * greedy_old + LEARNINGRATE * (
                    reward + DEFAULT_DISCOUNT * max_q_old)

    def select_action(self, state):
        """
        Returns an action, selected based on the current state
        """
        # If the state is not yet in qtable we need to add it with a reward of 0 for every action
        if state not in self.qtable:
            self.qtable[state] = [0] * self.num_actions
        # If random is less than epsilon we will explore
        if random.random() < self.epsilon:
            # Create a list for actions with a positive and negative reward and add the actions to these
            positive = []
            negative = []
            for i in range(4):
                action_score = self.qtable[state][i]
                if action_score >= 0:
                    positive.append(i)
                if action_score < 0:
                    negative.append(i)
            chance = random.uniform(0, 1)
            # With a 65% chance we pick a random positive action if there are any
            # Else we pick a negative chance if there are any
            # And otherwise we just pick a random action
            if chance < 0.65 and len(positive) > 0:
                return random.choice(positive)
            elif len(negative) > 0:
                return random.choice(negative)
            else:
                return random.randint(0, 3)
        # Else we will get the best action in this state
        # In case there are multiple best states we pick one of those at random
        # Otherwise it will always favor the lower indices which is bad in case the state has all 0 rewards
        else:
            score = -float("inf")
            action = []
            for i in range(4):
                action_score = self.qtable[state][i]
                if action_score == score:
                    action.append(i)
                if action_score > score:
                    score = action_score
                    action = [i]
            return random.choice(action)

    def report(self):
        """
        Function to print useful information, printed during the main loop
        """
        print("---")
