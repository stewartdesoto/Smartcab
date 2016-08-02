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

        # original values 0.1, 0.1, 0.5 work pretty well
        self.iteration = 0
        self.q = {}          # our q matrix is stored as a dict, q[State][Action] = long term utility (immediate reward + discounted reward)
        self.alpha = 0.1     # learning rate, q matrix is 1-alpha times the prior value
        self.gamma = 0.1     # discount factor, trades off the importance of immediate (r) versus later (q) rewards
        self.epsilon = 0.1   # tradeoff of exploration vs. exploitation, epsilon (greedy) is probability action is chosen at random

        self.state = None
        self.prior_state = None
        self.prior_action = None
        self.prior_reward = 0

        self.all_rewards = {}
        self.all_trial_moves = []
        self.move_count = 0
        self.optimal_distances = []

        self.successes = []

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

        location = self.env.agent_states[self]["location"] 
        destination = self.env.agent_states[self]["destination"]
        optimal_distance = self.env.compute_dist(location, destination)
        self.optimal_distances.append(optimal_distance)
        
        self.epsilon = 1./(self.iteration + 1)  # best is to have both this line
        self.alpha = self.epsilon               # and this line

        # import math
        # self.epsilon = 1./(math.log(float(self.iteration) + 2.))

        print "successes = {}".format(self.successes)
        print "optimal_distances = {}".format(self.optimal_distances)
        self.successes.append(0)
        if self.iteration > 0:
            self.all_trial_moves.append(self.move_count - 1)
            print "self.all_trial_moves = {}".format(self.all_trial_moves)
        self.move_count = 0
        self.iteration += 1
        self.all_rewards[self.iteration] = []
        print "q={}".format(self.q)

    def update(self, t):

        self.move_count += 1
        # Gather inputs, starter code section
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)


        possible_actions = [None, 'forward', 'left', 'right']

        # TODO: Update state
        # choose one of the following states
        # state space should be as small as possible for quick q learning

        # self.state = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['right'], inputs['left']) #
        self.state = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left']) # best one
        # self.state = (inputs['light'], inputs['oncoming'], inputs['right'], inputs['left'])
        # self.state = (self.next_waypoint)
        
        # if new state, initialize q matrix for this state and all possible actions, i.e. q[state][all possible actions] = 0
        if self.state not in self.q:
            self.q[self.state] = {}
            for try_action in possible_actions:
                self.q[self.state][try_action] = 0

        if self.prior_state:
            # Q update equation:
            #Q(state, action) = (1 - alpha(time)) * Q(state, action) + alpha(time) * (r + gamma * Q(next_state, action))

            # TODO: Select action according to your policy
           
            # find action in self.state which has largest q value
            maxq = -sys.float_info.max
            action_qmax = None
            for try_action in self.q[self.state]:
                if self.q[self.state][try_action] > maxq:
                    maxq = self.q[self.state][try_action]
                    action_qmax = try_action


            # update q value according to q update equation
            self.q[self.prior_state][self.prior_action] = (1 - self.alpha) * self.q[self.prior_state][self.prior_action]  +\
                                                     self.alpha * (self.prior_reward + self.gamma * self.q[self.state][action_qmax]  )


            # implement epsilon greedy algorithm for exploration vs. exploitation tradeoff
            rval = random.random() 
            if rval < self.epsilon:
                action = random.choice(possible_actions)
            else:
                action = action_qmax
        else: # first action in trial
            action = random.choice(possible_actions)
            # action = self.next_waypoint # optimal action, see how it performs

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.all_rewards[self.iteration].append(reward)
        if reward >= 10:
            self.successes[-1] = 1


        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        print "Deadline = {}".format(deadline)

         # set prior state equal to current state for next call to update
        self.prior_state = self.state # keep for next call to update
        self.prior_action = action
        self.prior_reward = reward

    def finish(self):
        print "Final data:"
        print "self.successes = {}".format(self.successes)
        print "self.all_rewards = {}".format(self.all_rewards)
        self.all_trial_moves.append(self.move_count - 1)
        print "self.all_trial_moves = {}".format(self.all_trial_moves)
        print "optimal_distances = {}".format(self.optimal_distances)

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

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    # print out final results
    a.finish()

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
