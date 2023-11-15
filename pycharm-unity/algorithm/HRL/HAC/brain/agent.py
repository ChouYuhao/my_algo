import torch
from torch import nn
from torch.nn import functional as F


# Policy Network that maps states to actions
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_offset, hidden_size=64):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_offset = action_offset
        self.layer1 = nn.Linear(self.state_dim * 2, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, self.action_dim)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.actor = nn.Sequential(
            self.layer1,
            self.relu,
            self.layer2,
            self.relu,
            self.layer3,
            self.tanh
        )

    def forward(self, state, subgoal):
        x = torch.cat([state, subgoal], 1)
        out = self.actor(x) + self.action_offset
        return out


# Q-value Network that maps state and action pairs to Q-values
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, max_horizon, hidden_size=64):
        super(Critic, self).__init__()
        self.max_horizon = max_horizon
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.layer1 = nn.Linear(self.state_dim * 2, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.critic = nn.Sequential(
            self.layer1,
            self.relu,
            self.layer2,
            self.relu,
            self.layer3,
            self.sigmoid
        )

    def forward(self, state, action, subgoal):
        # Qvalues are bounded in the range [− max_horizon, 0] because rewards used are nonpositive.
        # The bound of − max_horizon is
        # (i) helpful for learning Q-values as the critic function does not need to learn precise Q-values
        # for the large space of irrelevant actions in which the current state is far from the goal state.
        # (ii) ensures that subgoal states that were reached in hindsight should have higher Q-values than
        # any subgoal state that istoo distant and penalized during subgoal testing.
        x = torch.cat([state, action, subgoal], 1)
        out = self.critic(x) * self.max_horizon
        return out


class DDPG:
    def __init__(self, state_dim, action_dim, action_offset, max_horizon, lr):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_offset = action_offset
        self.max_horizon = max_horizon
        self.lr = lr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loss = nn.MSELoss()

        self.actor = Actor(self.state_dim, self.action_dim, self.action_offset).to(self.device)
        self.critic = Critic(self.state_dim, self.action_dim, self.max_horizon).to(self.device)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.loss_actor = None
        self.loss_critic = None

    def update_actor_critic(self, replay_buffer, n_iterations, batch_size):
        for i in range(n_iterations):
            state, action, reward, next_state, subgoal, discount, done = replay_buffer.sample_experience(batch_size)

            # get next action from actor network
            next_action = self.actor(next_state, subgoal).detach()

            # compute target Qvalue using Qvalue for next state and action from critic network
            next_Qvalue = self.critic(next_state, next_action, subgoal).detach()
            target_Qvalue = reward + ((1 - done) * discount * next_Qvalue)

            # compute critic network loss and optimize its parameters to minimize the loss
            self.loss_critic = self.loss(self.critic(state, action, subgoal), target_Qvalue)
            self.optimizer_critic.zero_grad()
            self.loss_critic.backward()
            self.optimizer_critic.step()

            # compute actor network loss and optimize its parameters to minimize the loss
            self.loss_actor = -self.critic(state, self.actor(state, subgoal), subgoal).mean()
            self.optimizer_actor.zero_grad()
            self.loss_actor.backward()
            self.optimizer_actor.step()
