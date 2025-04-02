import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt


# ニューラルネットワークの定義
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# リプレイメモリ
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# DQNエージェント
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # ネットワーク
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 最適化
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        # リプレイメモリ
        self.replay_buffer = ReplayBuffer(10000)
        self.batch_size = 64

        # 探索パラメータ
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99

        # ターゲットネットワーク更新頻度
        self.target_update = 10

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)

        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.policy_net(state)
        return q_values.argmax().item()

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # バッチサンプリング
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # 現在のQ値
        current_q = self.policy_net(states).gather(1, actions)

        # 次のQ値（ターゲットネットワークから）
        next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + (1 - dones) * self.gamma * next_q

        # 損失計算と最適化
        loss = self.criterion(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # イプシロン減衰
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())


# メイン学習ループ
def train_dqn():
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    episodes = 1000
    rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # リプレイバッファに追加
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # 学習
            agent.train()

            state = next_state
            total_reward += reward

            if done:
                break

        # 目標ネットワーク更新
        if episode % agent.target_update == 0:
            agent.update_target()

        rewards.append(total_reward)

        if episode % 10 == 0:
            avg_reward = np.mean(rewards[-10:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    # 学習したモデルを保存
    agent.save('dqn_cartpole.pth')

    # 学習曲線のプロット
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.plot(np.convolve(rewards, np.ones(10) / 10, mode='valid'), color='red')
    plt.title('DQN Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid()
    plt.savefig('dqn_learning_curve.png')
    plt.show()

    return agent


# 視覚的テスト実行
def test_agent(agent):
    env = gym.make('CartPole-v1', render_mode='human')
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Test Reward: {total_reward}")
    env.close()


# 実行
if __name__ == "__main__":
    agent = train_dqn()
    # テスト
    for _ in range(3):
        test_agent(agent)
