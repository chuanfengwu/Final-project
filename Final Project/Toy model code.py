#consulted chatgpt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Env
class ButtonLightEnv:
    def __init__(self):
        self.rules = [
            [0,1],
            [2],
            [1,3],
            [0,2,3]
        ]
        self.reset()

    def reset(self):
        self.state = np.random.randint(0,2,4).astype(np.float32)
        return self.state.copy()

    def step(self, action):
        new = self.state.copy()
        for l in self.rules[action]:
            new[l] = 1-new[l]
        r = 1 if new[0]==1 else -1
        self.state = new
        return new.copy(), r

# 2. Predictor
class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8,32),
            nn.ReLU(),
            nn.Linear(32,4),
            nn.Sigmoid()
        )
    def forward(self, s, a):
        return self.net(torch.cat([s,a],dim=1))

# agent planner
def plan(model, state):
    s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    best_a, best_r = None, -999
    for a in range(4):
        ao = torch.zeros(1,4)
        ao[0,a]=1
        ns = model(s,ao).detach().numpy()[0]
        est_r = 1 if ns[0]>0.5 else -1
        if est_r>best_r:
            best_r, best_a = est_r, a
    return best_a

# training
env = ButtonLightEnv()
model = Predictor()
opt = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCELoss()
loss_history = []

for step in range(2000):
    state = env.reset()
    for t in range(20):
        a = np.random.randint(0,4)
        ao = np.zeros(4); ao[a]=1
        ns,_ = env.step(a)
        s = torch.tensor(state).float().unsqueeze(0)
        ao_t = torch.tensor(ao).float().unsqueeze(0)
        ns_t = torch.tensor(ns).float().unsqueeze(0)
        pred = model(s,ao_t)
        loss = loss_fn(pred, ns_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_history.append(loss.item())
        state = ns

# test planner
env_test = ButtonLightEnv()
s = env_test.reset()
planner_rewards = 0
for t in range(50):
    a = plan(model, s)
    s,r = env_test.step(a)
    planner_rewards += r

# random baseline
env_rand = ButtonLightEnv()
s = env_rand.reset()
rand_rewards = 0
for t in range(50):
    a = np.random.randint(0,4)
    s,r = env_rand.step(a)
    rand_rewards += r

# plotting
plt.figure()
plt.plot(loss_history)
plt.title("Training Loss Curve")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.show()

plt.figure()
plt.bar(["Planner Agent", "Random Policy"], [planner_rewards, rand_rewards])
plt.title("Reward Comparison over 50 Steps")
plt.ylabel("Total Reward")
plt.show()

planner_rewards, rand_rewards
