"""agent.py: Contains the entire deep reinforcement learning agent."""
__author__ = "Erik Gärtner"

from collections import deque

import tensorflow as tf
import numpy as np

from .expreplay import ExpReplay


class Agent():
    """
    The agent class where you should implement the vanilla policy gradient agent.
    """

    def __init__(self, tf_session, state_size=(4,), action_size=2,
                 learning_rate=1e-3, gamma=0.99, memory_size=5000):
        """
        The initialization function. Besides saving attributes we also need
        to create the policy network in Tensorflow that later will be used.
        """

        n_hidden = 4
        n_outputs = action_size
        initializer = tf.contrib.layers.variance_scaling_initializer()

        self.state_size = state_size
        self.action_size = action_size
        self.tf_sess = tf_session
        self.gamma = gamma
        self.replay = ExpReplay(memory_size)

        with tf.variable_scope('agent'):
            # Create tf placeholders, i.e. inputs into the network graph.
            self.X = tf.placeholder(tf.float32,shape=[None,4])

            self.rewards = tf.placeholder(tf.float32)
            self.ep_actions = tf.placeholder(tf.int32)

            # Create the hidden layers
            hidden = tf.layers.dense(self.X, n_hidden,activation=tf.nn.elu,
                                     kernel_initializer=initializer)
            logits = tf.layers.dense(hidden,n_outputs,
                                     kernel_initializer=initializer)

            log_actions_prop = tf.log(tf.nn.softmax(logits))

            self.sample = tf.reshape(tf.multinomial(logits, 1), [])

            rewards = tf.reshape(self.rewards,[1,10])

            # Create the loss. We need to multiply the reward with the
            # log-probability of the selected actions.
            loss = -tf.reduce_sum(tf.matmul(log_actions_prop,rewards))

            # Create the optimizer to minimize the loss
            self.train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

            pass

        tf_session.run(tf.global_variables_initializer())

    def take_action(self, state):
        """
        Given the current state sample an action from the policy network.
        Return a the index of the action [0..N).
        """
        action = self.tf_sess.run(self.sample, feed_dict={self.X: state.reshape(1, len(state))})
        return action

    def record_action(self, state0, action, reward, state1, done):
       # arr = [state0,action,reward,state1]
        arr = (state0, action, reward, state1)

        self.replay.add(arr)

        """
        Record an action taken by the action and the associated reward
        and next state. This will later be used for traning.
        """
        pass

    def train_agent(self):
        """
        Train the policy network using the collected experiences during the
        episode(s).
        """
        # Retrieve collected experiences from memory
        experiences = np.array(self.replay.get_all())
       # rewards = np.array([h['reward'] for h in experiences])
        #rewards = experiences[:,2]
        rewards = np.array([r[2] for r in experiences])

        # Discount and normalize rewards
        norm_rewards = self.discount_rewards_and_normalize(rewards)

        # Shuffle for better learning
        shuffled_experiences = np.random.shuffle(experiences)

        # Feed the experiences through the network with rewards to compute and
        # minimize the loss.

        feed={
            self.X: [r[0] for r in experiences],
            self.rewards:norm_rewards,
            self.ep_actions:experiences[:,1]
        }
        self.tf_sess.run(self.train,feed_dict=feed)

        pass

    def discount_rewards_and_normalize(self, rewards):
        """
        Given the rewards for an epsiode discount them by gamma.
        Next since we are sending them into the neural network they should
        have a zero mean and unit variance.

        Return the new list of discounted and normalized rewards.
        """
        discounted_rewards = np.empty(len(rewards))
        cumulative_rewards = 0

        for step in reversed(range(len(rewards))):
            cumulative_rewards = rewards[step] + cumulative_rewards * self.gamma
            discounted_rewards[step] = cumulative_rewards

        reward_mean = discounted_rewards.mean()
        reward_std = discounted_rewards.std()

        return [(reward - reward_mean) / reward_std
                for reward in discounted_rewards]
