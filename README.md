# 🐍 Snake Game AI — Deep Q-Learning with PyTorch & Pygame

An implementation of a self-learning Snake agent using **Deep Q-Learning (DQN)**, built from scratch with **PyTorch** (neural network) and **Pygame** (game engine). The agent learns to play Snake purely through trial and error — no hardcoded rules.

---

## 📁 Project Structure

```
snake_game_ai/
│
├── agent.py              # AI agent: state extraction, memory, action selection, training loop
├── game.py               # Snake game environment for the AI (SnakeGameAI)
├── model.py              # Neural network (Linear_QNet) and Q-learning trainer (QTrainer)
├── helper.py             # Live training plot (score & mean score over games)
├── snake_game_human.py   # Playable Snake game for humans (keyboard-controlled)
├── arial.ttf             # Font file used by Pygame for score display
├── model/
│   └── model.pth         # Saved model weights (auto-generated after training)
└── README.md             # This file
```

---

## 🧠 How It Works

The agent uses **Deep Q-Learning**, a reinforcement learning technique where a neural network learns to predict the best action given the current game state.

### State (11 features)
At each step, the agent observes an 11-value boolean state vector:
- **Danger ahead / right / left** (3 values) — whether moving straight, turning right, or turning left causes a collision
- **Current direction** (4 values) — LEFT, RIGHT, UP, DOWN
- **Food location relative to head** (4 values) — food is left/right/up/down of head

### Actions (3 choices)
- `[1, 0, 0]` → Go straight
- `[0, 1, 0]` → Turn right
- `[0, 0, 1]` → Turn left

### Rewards
| Event | Reward |
|---|---|
| Eat food | +10 |
| Die (collision / timeout) | −10 |
| Everything else | 0 |

### Q-Learning Update (Bellman Equation)
```
Q_new = reward + gamma * max(Q(next_state))
```
- `gamma = 0.9` (discount factor)
- Loss: Mean Squared Error between predicted Q-values and target Q-values

---

## 📄 File Descriptions

### `agent.py`
The core training loop and agent logic.

- **`Agent` class**
  - `get_state(game)` — Encodes the 11-feature state vector from the current game
  - `remember(...)` — Stores `(state, action, reward, next_state, done)` in a replay memory deque (max 100,000 entries)
  - `train_short_memory(...)` — Trains on the single most recent step
  - `train_long_memory()` — Samples a random batch of 1,000 from memory and trains (experience replay)
  - `get_action(state)` — Epsilon-greedy action selection: explores randomly for the first ~80 games, then exploits the model
- **`train()` function** — Main loop: plays games endlessly, trains after each step and each game, saves the model on new high scores, and plots live progress

### `game.py`
The Snake environment designed for the AI agent (`SnakeGameAI`).

- Grid size: **640×480**, block size: **20px**, speed: **40 FPS**
- **`reset()`** — Resets snake to center, clears score, places new food
- **`play_step(action)`** — Executes one action, checks collisions, updates score, returns `(reward, game_over, score)`
- **`is_collision(pt)`** — Returns `True` if point hits a wall or the snake's own body
- **Timeout** — Game ends if `frame_iteration > 100 × snake_length` (prevents infinite loops)
- **`_move(action)`** — Converts a one-hot action `[straight, right, left]` into an absolute direction using a clockwise direction list

### `model.py`
Neural network and Q-trainer.

- **`Linear_QNet`** — A 2-layer fully-connected network:
  - Input: 11 neurons (state size)
  - Hidden: 256 neurons with ReLU activation
  - Output: 3 neurons (Q-value for each action)
  - `save()` — Saves model weights to `./model/model.pth`
- **`QTrainer`**
  - Optimizer: **Adam** (`lr=0.001`)
  - Loss: **Mean Squared Error (MSELoss)**
  - `train_step(...)` — Computes target Q-values using the Bellman equation and backpropagates the loss

### `helper.py`
Live training visualization using Matplotlib.

- **`plot(scores, mean_scores)`** — Renders a real-time line chart of per-game scores and running mean. Uses IPython display for inline notebook-style updating.

### `snake_game_human.py`
A standalone, keyboard-controlled version of Snake for human play.

- **`SnakeGame` class** — Same game logic as `game.py` but controlled via arrow keys
- Speed: **20 FPS** (slower than the AI version)
- No reward system — purely for human play and testing
- Run directly: `python snake_game_human.py`

### `arial.ttf`
Font file (Arial, ~1 MB) used by Pygame to render the score display on the game window. Both `game.py` and `snake_game_human.py` load this font.

### `model/model.pth`
Auto-generated when the AI beats its previous high score. Contains the saved PyTorch model weights (`state_dict`). Delete this file to train the agent from scratch.

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pygame torch numpy matplotlib ipython
```

### Train the AI Agent
```bash
python agent.py
```
- The game window opens automatically
- A live score plot updates after each game
- Best model is saved to `model/model.pth` whenever a new high score is achieved
- Training progress is printed to the console: `Game <N>  Score <S>  Record: <R>`

### Play Snake Yourself
```bash
python snake_game_human.py
```
Use arrow keys to control the snake. The game ends on wall or self-collision.

---

## 🏗️ Architecture Overview

```
agent.py  ──────────────────────────────────────────────────────┐
  Agent.get_state()  ──►  game.py (SnakeGameAI)                 │
  Agent.get_action() ──►  model.py (Linear_QNet)                │
  Agent.train_*()    ──►  model.py (QTrainer)                   │
  train() loop       ──►  helper.py (plot)                      │
                          model/model.pth (saved weights)       │
└───────────────────────────────────────────────────────────────┘

snake_game_human.py  ──►  (standalone, no AI dependency)
```

---

## 📊 Hyperparameters

| Parameter | Value |
|---|---|
| Max replay memory | 100,000 |
| Batch size | 1,000 |
| Learning rate | 0.001 |
| Discount factor (γ) | 0.9 |
| Hidden layer size | 256 |
| State size | 11 |
| Action space | 3 |
| Exploration threshold | 80 games |

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
