import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

import sys

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here

        # from collections import defaultdict
        self.q = {}          # our q matrix is stored as a dict, q[State][Action] = long term utility (immediate reward + discounted reward)
        self.alpha = 0.1     # learning rate, q matrix is 1-alpha times the prior value
        self.gamma = 0.1     # discount factor, trades off the importance of immediate (r) versus later (q) rewards
        self.epsilon = 0.5   # tradeoff of exploration vs. exploitation, epsilon (greedy) is probability action is chosen at random

        self.state = None
        self.prior_state = None
        self.prior_action = None
        self.prior_reward = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs, starter code section
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        possible_actions = [None, 'forward', 'left', 'right']

        # TODO: Update state
        # state space should be as small as possible for quick q learning
        self.state = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['right'], inputs['left'])
        
        # if new state, initialize q matrix this state and all possible actions q[state][all possible actions] = 0
        if self.state not in self.q:
            self.q[self.state] = {}
            for try_action in possible_actions:
                self.q[self.state][try_action] = 0

        print 
        print "calling update()"
        print "inputs = {}".format(inputs)
        print "self.next_waypoint = {}".format(self.next_waypoint)
        print 'self.state={}'.format(self.state)
        print "self.q = {}".format(self.q)
        print 'self.prior_state={}'.format(self.prior_state)
        print 'self.prior_action={}'.format(self.prior_action)

        # choose action here
        if self.prior_state:
            #Q(state, action) = (1 - alpha(time)) * Q(state, action) + alpha(time) * (r + gamma * Q(next_state, action))
            if self.prior_state in self.q and self.prior_action in self.q[self.prior_state]:
                self.q[self.prior_state][self.prior_action] = (1 - self.alpha) * self.q[self.prior_state][self.prior_action]  +\
                                                     self.alpha * (self.prior_reward + self.gamma * self.q[self.state][self.prior_action]  )

            else:

                if self.prior_state in self.q:
                    self.q[self.prior_state][self.prior_action] = self.alpha * self.prior_reward
                else:
                    self.q[self.prior_state] = {}
                    self.q[self.prior_state][self.prior_action] = self.alpha * self.prior_reward

            # TODO: Select action according to your policy

            # find action with max q value
            print "finding max q"
            print "self.q = {}".format(self.q)
            maxq = -sys.float_info.max
            action_qmax = None

            if self.state in self.q:
                for try_action in self.q[self.state]:
                    if self.q[self.state][try_action] > maxq:
                        maxq = self.q[self.state][try_action]
                        action_qmax = try_action


            rval = random.random() 
            print "rval = {}".format(rval)
            if rval < self.epsilon:
                action = random.choice(possible_actions)
                print "greedy choice"
            else:
                action = action_qmax
                print "qmax choice"
        else: # first action in trial
            action = random.choice(possible_actions)


        print "chosen action = {}".format(action)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        
        # set prior state equal to current state for next call to update

        self.prior_state = self.state # keep for next call to update
        self.prior_action = action
        self.prior_reward = reward

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=20)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()


        #action = None
        # import random
        # possible_actions = [None]
        # if inputs['light'] == 'green':
        #     possible_actions.extend(['forward', 'right'])
        #     if inputs['oncoming'] == None:
        #         possible_actions.extend(['left'])
        # elif inputs['light'] == 'red' and inputs['left'] == None:
        #     possible_actions.extend(['right'])
