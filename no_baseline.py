import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import datetime

device = torch.device('cpu')

mode_labels = [
    'BAD, no CF gradient',
    'BAD, with CF gradient',
    'Vanilla PG',
]

payoff_values = np.asarray([
    [
        [[10, 0, 0], [4, 8, 4], [10, 0, 0]],
        [[0, 0, 10], [4, 8, 4], [0, 0, 10]],
    ],
    [
        [[0, 0, 10], [4, 8, 4], [0, 0, 0]],
        [[10, 0, 0], [4, 8, 4], [10, 0, 0]],
    ],
], dtype=np.float32)
payoff_tensor = torch.tensor(payoff_values)

num_cards = payoff_values.shape[0] # C
num_actions = payoff_values.shape[-1] # A

# Number of hidden layers in A1.p1 and baselines
num_hidden = 32


# Agent 0
class A0(nn.Module):
    def __init__(self):
        super(A0, self).__init__()
        self.weights_0 = nn.Linear(num_cards, num_actions)

    def forward(self, cards_0):
        """
        Args:
          cards_0 of shape (batch_size,): Input is a single card 0,...,C dealt to player0
        Returns:
          u0 of shape (batch_size,): Action 0,...,A of player0
          beliefs of shape (batch_size, C):
            Probability distribution of card_0 based on cf_action (policy) and u0 (action)
            [0.5 0.5] when policy for different cards is the same
            [1 0] or [0 1] when action uniquely defines card (policy for different cards is different)
          log_cf of shape (batch_size, C): Log prob of the action chosen for each possible card_0
        """

        # These are the 'counterfactual inputs', i.e., all possible cards.
        # repeated_in.shape = (batch_size * C,)
        repeated_in = torch.tensor([i % num_cards
                                    for i in range(num_cards * batch_size)])
        one_hot = F.one_hot(repeated_in, num_cards).float() # (batch_size * C, C)

        # Next we calculate the counterfactual action and log_p for each batch and hand.

        # Log prob of choosing action when given input c
        # log_p0.shape = (batch_size * C, A)
        log_p0 = F.log_softmax(self.weights_0(one_hot), dim=1)

        # Multinomial distribution with num of trials 1, event log probabilities log_p0
        m = torch.distributions.multinomial.Multinomial(1, logits=log_p0)
        cf_action = m.sample() # one-hot matrix of shape (batch_size * C, A)
        # Log prob of choosing that action. log_cf.shape = (batch_size * C,)
        log_cf = m.log_prob(cf_action)

        # Reverse one-hot encoding. cf_action.shape = (batch_size * C,)
        cf_action = torch.argmax(cf_action, dim=1)
        # Some reshaping
        cf_action = cf_action.reshape([batch_size, -1]) # (batch_size, C)
        log_cf = log_cf.reshape([batch_size, -1]) # (batch_size, C)

        # Now we need to know the action the agent actually took.
        # This is done by indexing the cf_action with the private observation.
        # u0.shape = (batch_size,)
        u0 = cf_action[range(batch_size), cards_0]

        # Repeating the action chosen so that we can check all matches
        # repeated_actions.shape = (batch_size, C)
        repeated_actions = u0.repeat(num_cards, 1).transpose(1, 0)

        # A hand is possible iff the action in that hand matches the action chosen.
        weights = (repeated_actions == cf_action).float() # (batch_size, C)

        # Normalize beliefs to sum to 1.
        beliefs = weights / weights.sum(dim=-1, keepdim=True) # (batch_size, C)

        return u0, beliefs, log_cf


# Agent 1
class A1(nn.Module):
    def __init__(self):
        super(A1, self).__init__()
        # Inputs: one_hot encoded actions of player0, cards of player1
        p1_input_size = num_actions + num_cards
        # BAD agent also takes in beliefs
        if bad_mode != 2:
            p1_input_size += num_cards

        self.p1 = nn.Sequential(
            nn.Linear(p1_input_size, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_actions))

    def forward(self, joint_in1):
        """
        Arguments:
          joint_in1 of shape
            - Vanilla PG: (batch_size, A + C):  action taken by player0,
                                                card dealt to player1

            - BAD: (batch_size, A + C + C):     action taken by player0,
                                                beliefs about the card of player0,
                                                card dealt to player1
        Returns:
          u1 of shape (batch_size,): Action 0,...,A of player1
          log_p1 of shape (batch_size,): Log prob of action chosen
        """

        # Evaluate policy for agent 1
        # Log probs of actions. log_p1.shape = (batch_size, A)
        log_p1 = F.log_softmax(self.p1(joint_in1), 1)

        # Sample and get log-prob of action selected
        m = torch.distributions.multinomial.Multinomial(1, logits=log_p1)
        u1 = m.sample()  # one_hot (batch_size, A)
        log_p1 = m.log_prob(u1) # (batch_size,)

        # Reverse the one-hot encoding
        u1 = torch.argmax(u1, dim=1) # (batch_size,)
        return u1, log_p1


class Baseline_0_mlp(nn.Module):
    def __init__(self):
        super(Baseline_0_mlp, self).__init__()
        self.mlp = nn.Sequential(
            # Inputs: one_hot encoded cards_0, cards_1
            nn.Linear(2 * num_cards, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 1))

    def forward(self, x):
        return self.mlp(x)


class Baseline_1_mlp(nn.Module):
    def __init__(self):
        super(Baseline_1_mlp, self).__init__()
        # Inputs: one_hot encoded cards_0, u0, cards_1
        baseline_1_input_size = num_cards + num_actions + num_cards
        # BAD baseline also takes in beliefs
        if bad_mode != 2:
            baseline_1_input_size += num_cards

        self.mlp = nn.Sequential(
            nn.Linear(baseline_1_input_size, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 1))

    def forward(self, x):
        return self.mlp(x)



def train(bad_mode,
          #batch_size=32,
          num_runs=1,
          num_episodes=5000,
          num_readings=100,
          seed=42,
          debug=False):
    np.random.seed(seed)
    torch.manual_seed(seed)

    rewards = np.zeros((num_runs, num_readings + 1))
    interval = num_episodes // num_readings

    for run_id in range(num_runs):
        if run_id % max(num_runs // 10, 1) == 0:
            print('Run {}/{} ...'.format(run_id + 1, num_runs))

        a0 = A0()
        a1 = A1()

        # baseline_0_mlp = Baseline_0_mlp()
        # baseline_1_mlp = Baseline_1_mlp()

        policy_parameters = list(a0.parameters()) + list(a1.parameters())
        adam_policy = torch.optim.Adam(policy_parameters)

        # baseline_parameters = list(baseline_0_mlp.parameters()) + list(baseline_1_mlp.parameters())
        # adam_baseline = torch.optim.Adam(baseline_parameters)

        for episode_id in range(num_episodes + 1):
            adam_policy.zero_grad()

            cards_0 = np.random.choice(num_cards, size=batch_size)
            cards_1 = np.random.choice(num_cards, size=batch_size)
            input_0 = torch.tensor(cards_0)
            input_1 = torch.tensor(cards_1)

            u0, beliefs, log_cf = a0(cards_0) # TODO change to input_0

            if bad_mode == 2:
                joint_in1 = torch.cat((F.one_hot(u0, num_actions).float(),
                                       F.one_hot(input_1, num_cards).float()), 1)
            else:
                joint_in1 = torch.cat((F.one_hot(u0, num_actions).float(),
                                       beliefs,
                                       F.one_hot(input_1, num_cards).float()), 1)
            u1, log_p1 = a1(joint_in1)

            batch_reward = torch.stack([payoff_tensor[input_0[i], input_1[i], u0[i], u1[i]]
                                   for i in range(batch_size)]) # (batch_size,)

            mean_batch_reward = torch.mean(batch_reward).item()

            # log_cf contains log probabilities for counterfactual actions
            # we index those probabilities with cards_0 to find the log prob for the action
            # that agent 0 actually took
            log_p0 = log_cf[range(batch_size), cards_0]
            # Joint-action includes all the counterfactual probs - it's simply the sum.
            joint_log_p0 = log_cf.sum(dim=-1)

            # Log-prob used for learning.
            if bad_mode == 1:
                log_p0_train = joint_log_p0
            else:
                log_p0_train = log_p0
            log_p1_train = log_p1

            # Policy-gradient loss
            pg_final = torch.mean((batch_reward) * log_p0_train)
            pg_final += torch.mean((batch_reward) * log_p1_train)
            pg_final = -pg_final # minimize negative

            pg_final.backward()
            adam_policy.step()

            # Maybe save.
            if episode_id % interval == 0:
                rewards[run_id, episode_id // interval] = mean_batch_reward

            # Maybe log.
            if debug and episode_id % (num_episodes // 5) == 0:
                print(episode_id, 'reward:', mean_batch_reward)

    return rewards


debug = False
if debug:
    num_runs = 10
    num_episodes = 3000
else:
    num_runs = 10
    num_episodes = 15000
num_readings = 100
batch_size=32

skip_training = True
if not skip_training:
    rewards_by_bad_mode = {}
    for bad_mode in range(3):
        print('Running', mode_labels[bad_mode])
        rewards_by_bad_mode[bad_mode] = train(bad_mode,
                                              num_runs=num_runs,
                                              num_episodes=num_episodes,
                                              num_readings=num_readings,
                                              debug=debug)
        print('')

    save_rewards = True
    if save_rewards:
        time = datetime.datetime.now().strftime("%m%d%H%M%S")
        filename = "logs/no_baseline" + time + ".npy"
        np.save(filename, rewards_by_bad_mode)

if skip_training:
    rewards_by_bad_mode = np.load('logs/no_baseline.npy', allow_pickle=True).item()
    num_runs = rewards_by_bad_mode[0].shape[0]
    num_readings = rewards_by_bad_mode[0].shape[1] - 1
    # can't access num_episodes!
    num_episodes = 15000

plt.figure(figsize=(10, 5))

save_every = num_episodes // num_readings
steps = np.arange(num_readings + 1) * save_every
for bad_mode in range(3):
  rewards = rewards_by_bad_mode[bad_mode]
  mean = rewards.mean(axis=0)
  sem = rewards.std(axis=0) / np.sqrt(num_runs)
  plt.plot(steps, mean, label=mode_labels[bad_mode])
  plt.fill_between(steps, mean - sem, mean + sem, alpha=0.3)
plt.ylim(5, 10)
plt.legend()
plt.show()
