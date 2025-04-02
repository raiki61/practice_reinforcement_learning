import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import time


# 元のDQNモデル構造を定義（モデルを正しく読み込むために必要）
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


# 保存されたモデルを使用するためのシンプルなエージェント
class LoadedDQNAgent:
    def __init__(self, model_path, state_dim, action_dim):
        self.policy_net = DQN(state_dim, action_dim)
        self.policy_net.load_state_dict(torch.load(model_path))
        self.policy_net.eval()  # 評価モードに設定

    def act(self, state):
        with torch.no_grad():  # 勾配計算不要
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state)
            return q_values.argmax().item()


def test_saved_model(model_path, episodes=5, render_delay=0.02):
    """
    保存されたDQNモデルを読み込み、CartPole環境でテストする

    引数:
        model_path: 保存されたモデルファイルのパス
        episodes: テストするエピソード数
        render_delay: 各ステップ間の遅延（秒）
    """
    # 環境の初期化（レンダリングモード指定）
    env = gym.make('CartPole-v1', render_mode='human')

    # 状態・行動空間の次元を取得
    state_dim = env.observation_space.shape[0]  # CartPoleでは4
    action_dim = env.action_space.n  # CartPoleでは2

    # エージェントの初期化
    agent = LoadedDQNAgent(model_path, state_dim, action_dim)

    print(f"モデル {model_path} を読み込みました。{episodes}エピソード実行します...")

    # 各エピソードを実行
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        print(f"エピソード {episode + 1} 開始...")

        while not done:
            # モデルを使用して行動を選択
            action = agent.act(state)

            # 環境で行動を実行
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1

            # 視覚化のための遅延
            time.sleep(render_delay)

        print(f"エピソード {episode + 1}: 報酬={total_reward}, ステップ数={steps}")

    env.close()
    print("テスト完了")


if __name__ == "__main__":
    # モデルファイルのパス
    model_path = "dqn_cartpole.pth"

    # テストの実行
    test_saved_model(model_path)
