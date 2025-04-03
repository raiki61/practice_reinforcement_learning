import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau


# 強化版DuelingDQNアーキテクチャ（過学習対策と表現力向上）
class EnhancedDuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units=[64, 128, 64]):
        super(EnhancedDuelingDQN, self).__init__()

        # 詳細な重み初期化
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=1.0)  # 直交初期化で勾配の安定性向上
                torch.nn.init.constant_(m.bias, 0.1)  # 小さな正の値でReLUの死活を防止

        # ノイズ付き入力層（わずかなノイズを加えて頑健性向上）
        self.noisy_input = lambda x: x + torch.randn_like(x) * 0.1 if self.training else x

        # 特徴抽出層（より深く複雑なアーキテクチャ）
        layers = []
        in_features = state_dim

        for units in hidden_units:
            layers.extend([
                nn.Linear(in_features, units),
                nn.LayerNorm(units),  # バッチ間の一貫性のためLayerNormを使用
                nn.ReLU(),
                nn.Dropout(0.1)  # 過学習防止のためのドロップアウト
            ])
            in_features = units

        self.feature_layer = nn.Sequential(*layers)

        # 状態価値ストリーム（シンプルだが効果的）
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_units[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # アドバンテージストリーム（より複雑に）
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_units[-1], 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

        # 重み初期化の適用
        self.apply(init_weights)

    def forward(self, x):
        # トレーニング時のみノイズを追加
        if self.training:
            x = x + torch.randn_like(x) * 0.01

        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # 数値的安定性のための改良版Duelingアーキテクチャ
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


# 効率的なPER（優先度付き経験再生）実装
class EnhancedPrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_frames=100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha  # 優先度の指数
        self.beta_start = beta_start  # 重要度サンプリング開始値
        self.beta_end = beta_end  # 重要度サンプリング終了値
        self.beta_frames = beta_frames  # ベータ増加フレーム数
        self.frame_idx = 0  # 現在のフレーム
        self.epsilon = 1e-5  # 数値安定性のための小さな定数

        self.max_priority = 1.0

        # 経験の保存期間追跡（古い経験の優先度減衰用）
        self.experience_age = {}

    def push(self, state, action, reward, next_state, done):
        # 新経験には最大優先度を付与
        max_priority = self.max_priority

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        # 経験を保存
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority

        # 経験の年齢をリセット
        self.experience_age[self.position] = 0

        # すべての経験の年齢を増加
        for idx in self.experience_age:
            if idx != self.position:
                self.experience_age[idx] += 1

        self.position = (self.position + 1) % self.capacity
        self.frame_idx += 1

    def get_beta(self):
        # ベータ値を徐々に増加（重要度サンプリングの影響を徐々に強く）
        return min(self.beta_end,
                   self.beta_start + (self.beta_end - self.beta_start) * self.frame_idx / self.beta_frames)

    def age_based_priority_decay(self):
        # 古い経験の優先度を徐々に下げる（1000フレームごとに実行）
        if self.frame_idx % 1000 == 0 and self.buffer:
            for idx, age in self.experience_age.items():
                if age > 5000:  # 5000フレーム以上経過した経験
                    decay_factor = 0.99  # 徐々に減衰
                    self.priorities[idx] *= decay_factor

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        # 古い経験の優先度減衰を適用
        self.age_based_priority_decay()

        # 現在のベータ値を取得
        beta = self.get_beta()

        # 優先度に基づく確率計算
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # 系統的サンプリング（多様性確保）
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]

        # 重要度サンプリング重みの計算
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            if hasattr(priority, "__len__"):
                priority = priority[0]

            # エラーの二乗を使用（大きなエラーをより強調）
            priority = float(priority) ** 2 + self.epsilon
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)


# N-step経験再生と適応的探索を備えたDQNエージェント
class OptimizedDQNAgent:
    def __init__(self, state_dim, action_dim, device="cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # ネットワークアーキテクチャを改良
        self.policy_net = EnhancedDuelingDQN(state_dim, action_dim).to(device)
        self.target_net = EnhancedDuelingDQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 評価モードに固定

        # 最適化器と学習率スケジューラ
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0005, eps=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5,
                                           patience=5, verbose=True, min_lr=0.00005)

        # N-step学習のバッファ（CartPoleでは5stepが最適）
        self.n_steps = 5
        self.n_step_buffer = deque(maxlen=self.n_steps)

        # 強化されたリプレイメモリ
        self.replay_buffer = EnhancedPrioritizedReplayBuffer(100000)
        self.batch_size = 256  # より大きなバッチサイズで勾配の安定性向上

        # 最適化された探索戦略
        self.epsilon = 1.0
        self.epsilon_decay = 0.998  # より緩やかな減衰
        self.epsilon_min = 0.1  # 最小値を高めに設定
        self.gamma = 0.99

        # ターゲットネットワーク更新パラメータ
        self.tau = 0.001  # ソフトアップデート係数

        # 学習の安定化パラメータ
        self.learn_counter = 0
        self.learn_frequency = 2  # 2ステップに1回学習（計算効率向上）

        # 統計情報
        self.losses = []

        # カリキュラム学習のためのステージ
        self.curriculum_stage = 0
        self.curriculum_thresholds = [50, 100, 200, 300]  # 各ステージの目標スコア

    def update_curriculum(self, avg_reward):
        # 現在の性能に基づいてカリキュラムステージを更新
        for i, threshold in enumerate(self.curriculum_thresholds):
            if avg_reward < threshold and i > self.curriculum_stage:
                return

        if avg_reward >= self.curriculum_thresholds[self.curriculum_stage] and self.curriculum_stage < len(
                self.curriculum_thresholds) - 1:
            self.curriculum_stage += 1
            print(
                f"Advancing to curriculum stage {self.curriculum_stage}: target reward {self.curriculum_thresholds[self.curriculum_stage]}")

            # 新ステージでは一時的に探索を増加
            self.epsilon = min(0.3, self.epsilon * 1.2)

    def update_epsilon(self, reward_history):
        # 過去の報酬に基づいて適応的に探索率を調整
        if len(reward_history) < 10:
            return

        avg_reward = np.mean(reward_history[-10:])
        best_avg = max([np.mean(reward_history[i:i + 10]) for i in
                        range(max(0, len(reward_history) - 50), len(reward_history) - 9)])

        # 性能低下時は探索を増加
        if avg_reward < 0.7 * best_avg:
            self.epsilon = min(0.3, self.epsilon * 1.5)
            print(f"Performance drop detected. Increasing exploration to epsilon={self.epsilon:.3f}")
        # 高性能時は探索を減少
        elif avg_reward > 450:
            self.epsilon = max(self.epsilon_min, self.epsilon * 0.9)

    def state_normalization(self, state):
        # 状態の正規化（より安定した学習のため）
        # CartPole特有のスケーリング
        state_means = np.array([0.0, 0.0, 0.0, 0.0])
        state_stds = np.array([2.4, 10.0, 0.2, 10.0])  # CartPoleの各次元の典型的な範囲
        return (state - state_means) / state_stds

    def act(self, state, evaluation=False):
        normalized_state = self.state_normalization(state)

        # 評価モードまたはランダム閾値を下回った場合は学習済み方策を使用
        if evaluation or np.random.rand() > self.epsilon:
            with torch.no_grad():
                self.policy_net.eval()
                state = torch.FloatTensor(normalized_state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                if not evaluation:
                    self.policy_net.train()
                return q_values.argmax().item()
        else:
            # ランダム行動（カリキュラムステージに応じて調整）
            if self.curriculum_stage > 2 and np.random.rand() < 0.7:
                # 高ステージでは半ランダム行動（過去の学習を活用）
                with torch.no_grad():
                    self.policy_net.eval()
                    state = torch.FloatTensor(normalized_state).unsqueeze(0).to(self.device)
                    q_values = self.policy_net(state).cpu().numpy().flatten()
                    # ソフトマックスでランダム性を入れつつ学習した価値も反映
                    probs = np.exp(q_values * 2) / np.sum(np.exp(q_values * 2))
                    return np.random.choice(self.action_dim, p=probs)
            return random.randrange(self.action_dim)

    def push_n_step(self, state, action, reward, next_state, done):
        # N-step経験を蓄積
        self.n_step_buffer.append((state, action, reward, next_state, done))

        if len(self.n_step_buffer) < self.n_steps and not done:
            return

        # N-step リターンの計算
        reward_n = 0
        gamma_n = 1
        nth_state = None
        nth_done = False

        for i, (s, a, r, ns, d) in enumerate(self.n_step_buffer):
            reward_n += gamma_n * r
            gamma_n *= self.gamma

            if d:
                nth_done = True
                nth_state = ns
                break

            if i == len(self.n_step_buffer) - 1:
                nth_state = ns
                nth_done = d

        # 最初の経験とN-step先の情報を組み合わせてバッファに追加
        first_experience = self.n_step_buffer[0]
        self.replay_buffer.push(first_experience[0], first_experience[1],
                                reward_n, nth_state, nth_done)

        # エピソード終了時はバッファをクリア
        if done:
            self.n_step_buffer.clear()

    def train(self):
        self.learn_counter += 1

        if len(self.replay_buffer) < self.batch_size or self.learn_counter % self.learn_frequency != 0:
            return 0

        # バッチサンプリング（優先度付き）
        transitions, indices, weights = self.replay_buffer.sample(self.batch_size)

        # バッチデータの準備と正規化
        states, actions, rewards, next_states, dones = zip(*transitions)

        # 状態の正規化
        states = np.array([self.state_normalization(s) for s in states])
        next_states = np.array([self.state_normalization(s) for s in next_states])

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        # Double DQN: ポリシーネットワークで次の行動を選択
        with torch.no_grad():
            self.policy_net.eval()
            next_q_values = self.policy_net(next_states)
            next_actions = next_q_values.max(1)[1].unsqueeze(1)
            self.policy_net.train()

        # ターゲットネットワークで次の行動の価値を評価
        with torch.no_grad():
            # N-stepの割引を考慮
            target_next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * (self.gamma ** self.n_steps) * target_next_q_values

        # 現在のQ値
        current_q = self.policy_net(states).gather(1, actions)

        # TD誤差計算（優先度更新用）
        td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()

        # Huber損失（外れ値に強い）
        loss = F.smooth_l1_loss(current_q, target_q, reduction='none')
        # 重要度サンプリング重みを適用
        loss = (loss * weights).mean()

        # L2正則化の追加（過学習防止）
        l2_reg = 0.0001
        for param in self.policy_net.parameters():
            loss += l2_reg * torch.sum(param ** 2)

        # 最適化ステップ
        self.optimizer.zero_grad()
        loss.backward()
        # 勾配クリッピング（爆発する勾配の防止）
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # 優先度の更新
        self.replay_buffer.update_priorities(indices, td_errors)

        # ターゲットネットワークのソフトアップデート
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)

        # イプシロン減衰（基本的な減衰のみ、適応的なものは別メソッドで処理）
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.losses.append(loss.item())
        return loss.item()

    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'curriculum_stage': self.curriculum_stage,
            'learn_counter': self.learn_counter
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epsilon = checkpoint['epsilon']
        self.curriculum_stage = checkpoint.get('curriculum_stage', 0)
        self.learn_counter = checkpoint.get('learn_counter', 0)


# 詳細な報酬整形関数
def shape_reward(state, reward, done, steps, max_steps=500):
    shaped_reward = reward

    if done and steps < max_steps:
        # 進行度に応じたペナルティ調整
        progress_factor = min(1.0, steps / 200.0)
        shaped_reward = -3.0 * (1.0 - progress_factor)
    else:
        # カートの位置と棒の角度に基づく追加報酬
        cart_position = abs(state[0])  # カートの中心からの距離
        pole_angle = abs(state[2])  # 棒の角度
        cart_velocity = abs(state[1])  # カートの速度
        pole_velocity = abs(state[3])  # 棒の角速度

        # 中心に近いほど、角度が小さいほど良い
        position_reward = 0.1 * (1.0 - min(1.0, cart_position / 2.4))
        angle_reward = 0.2 * (1.0 - min(1.0, pole_angle / 0.2))

        # 速度が小さいほど安定している（特に高ステップ数で重要）
        if steps > 300:
            velocity_reward = 0.05 * (1.0 - min(1.0, (cart_velocity + pole_velocity) / 10.0))
            shaped_reward += position_reward + angle_reward + velocity_reward
        else:
            shaped_reward += position_reward + angle_reward

    return shaped_reward


# 最適化された学習ループ
def train_optimized_dqn():
    # 再現性のための乱数シード設定
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make('CartPole-v1')
    env.reset(seed=seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # デバイスの選択
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    agent = OptimizedDQNAgent(state_dim, action_dim, device)
    episodes = 1000
    rewards = []
    avg_rewards = []
    best_avg_reward = -float('inf')
    patience_counter = 0
    patience_limit = 100  # より長い忍耐期間

    # 多様なシード値を使用するためのカウンター
    seed_counter = 0

    # 早期停止用の閾値（最大性能の90%でも良しとする）
    acceptable_performance_ratio = 0.9

    for episode in range(episodes):
        # 毎回異なるシード値を使用して多様な初期状態を確保
        current_seed = seed + seed_counter
        seed_counter += 1
        state, _ = env.reset(seed=current_seed)
        total_reward = 0
        episode_loss = 0
        steps = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            # 詳細な報酬整形の適用
            shaped_reward = shape_reward(state, reward, done, steps)

            # N-step経験をバッファに追加
            agent.push_n_step(state, action, shaped_reward, next_state, done)

            # 学習（頻度制限付き）
            if steps % agent.learn_frequency == 0:
                loss = agent.train()
                if loss:
                    episode_loss += loss

            state = next_state
            total_reward += reward

        rewards.append(total_reward)

        # 過去10エピソードの移動平均
        if len(rewards) >= 10:
            avg_reward = np.mean(rewards[-10:])
            avg_rewards.append(avg_reward)

            # カリキュラム学習の更新
            agent.update_curriculum(avg_reward)

            # 適応的探索率の更新
            agent.update_epsilon(rewards)

            # 学習率スケジューラの更新
            agent.scheduler.step(avg_reward)

            # 最良モデルの保存
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save('best_dqn_cartpole.pth')
                patience_counter = 0
                print(f"New best model saved with avg reward: {best_avg_reward:.1f}")
            else:
                patience_counter += 1

        # 学習進捗の出力
        if episode % 10 == 0:
            avg_10ep = np.mean(rewards[-10:] if len(rewards) >= 10 else rewards)
            print(f"Episode {episode}/{episodes}, Reward: {total_reward:.1f}, " +
                  f"Avg(10): {avg_10ep:.1f}, Epsilon: {agent.epsilon:.3f}, " +
                  f"Curriculum Stage: {agent.curriculum_stage}")

        # 適応的早期停止条件
        if patience_counter >= patience_limit and episode > 200:
            current_avg = np.mean(rewards[-10:])
            # 現在の性能が過去最高の90%以上ならまだ継続
            if current_avg < best_avg_reward * acceptable_performance_ratio:
                print(f"Early stopping at episode {episode}, no significant improvement for {patience_limit} episodes")
                print(f"Current avg: {current_avg:.1f}, Best avg: {best_avg_reward:.1f}")
                break
            else:
                # 性能が十分高ければカウンターをリセット
                print(f"Performance still acceptable at {current_avg:.1f}/{best_avg_reward:.1f}. Continuing training.")
                patience_counter = patience_limit // 2

        # 探索リセットの条件（低性能状態が続く場合）
        if patience_counter >= 30 and patience_counter % 30 == 0 and episode > 50:
            if np.mean(rewards[-10:]) < 200:  # 性能が悪い場合のみリセット
                old_epsilon = agent.epsilon
                agent.epsilon = min(0.5, agent.epsilon * 2.0)
                print(f"Resetting exploration at episode {episode} from {old_epsilon:.3f} to {agent.epsilon:.3f}")

        # 絶対的な成功条件: 平均報酬が490を超えて10エピソード継続
        if len(avg_rewards) >= 10 and all(r > 490 for r in avg_rewards[-10:]):
            print(f"Training successfully completed at episode {episode} with stable high performance!")
            break

    # 詳細な学習曲線
    plt.figure(figsize=(15, 12))

    # 報酬プロット
    plt.subplot(3, 1, 1)
    plt.plot(rewards, alpha=0.6, color='skyblue', label='Episode Reward')
    if avg_rewards:
        avg_indices = range(9, 9 + len(avg_rewards))
        plt.plot(avg_indices, avg_rewards, color='red', linewidth=2, label='10-Episode Average')

    # 重要なしきい値を水平線で表示
    plt.axhline(y=500, color='green', linestyle='--', alpha=0.7, label='Maximum Reward')
    plt.axhline(y=200, color='orange', linestyle='--', alpha=0.7, label='Basic Competence')
    plt.axhline(y=400, color='purple', linestyle='--', alpha=0.7, label='Advanced Competence')

    plt.title('Enhanced DQN Learning Progress', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)

    # 損失曲線
    plt.subplot(3, 1, 2)
    window_size = 100  # 移動平均ウィンドウ
    if len(agent.losses) > window_size:
        losses_smooth = np.convolve(agent.losses, np.ones(window_size) / window_size, mode='valid')
        plt.plot(range(len(losses_smooth)), losses_smooth, color='crimson')
    else:
        plt.plot(agent.losses, color='crimson')
    plt.title('Training Loss', fontsize=16)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, alpha=0.3)

    # イプシロン値の推移（探索の減衰を視覚化）
    plt.subplot(3, 1, 3)
    epsilon_values = [1.0]  # 初期値
    for i in range(1, episode + 1):
        if i % 10 == 0 and i // 10 < len(avg_rewards):
            epsilon_values.append(agent.epsilon_min + (1.0 - agent.epsilon_min) * np.exp(-0.01 * i))

    plt.plot(range(0, episode + 1, 10), epsilon_values, color='darkgreen', marker='o', markersize=3)
    plt.title('Exploration Rate (Epsilon) Decay', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Epsilon Value', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('optimized_dqn_learning_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 最終モデルを保存
    agent.save('final_dqn_cartpole.pth')
    print(f"Training completed. Best avg reward: {best_avg_reward:.1f}")

    return agent


# 詳細な評価関数
def test_optimized_agent(agent, num_episodes=10, render=True):
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    total_rewards = []
    episode_steps = []

    for i in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            # 評価モード（最適な方策を使用）
            action = agent.act(state, evaluation=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        total_rewards.append(total_reward)
        episode_steps.append(steps)
        if render:
            print(f"Test Episode {i + 1}/{num_episodes}: Steps={steps}, Reward={total_reward}")

    # 詳細な統計情報
    print(f"\n== Performance Evaluation ({num_episodes} episodes) ==")
    print(f"Mean Reward: {np.mean(total_rewards):.1f} ± {np.std(total_rewards):.1f}")
    print(f"Median Reward: {np.median(total_rewards):.1f}")
    print(f"Min/Max Reward: {np.min(total_rewards):.1f}/{np.max(total_rewards):.1f}")
    print(f"Mean Episode Length: {np.mean(episode_steps):.1f} steps")
    print(f"Success Rate (500 steps): {(sum(r == 500 for r in total_rewards) / num_episodes) * 100:.1f}%")

    env.close()
    return total_rewards


# 実行
if __name__ == "__main__":
    agent = train_optimized_dqn()
    # 最終評価
    test_optimized_agent(agent, num_episodes=5)
