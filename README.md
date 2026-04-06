# Text Flappy Bird - Reinforcement Learning Assignment

## 📋 Overview

This project implements and compares two reinforcement learning agents in the **TextFlappyBird-v0** environment:

1. **Monte Carlo Control** - On-policy, first-visit, ε-greedy algorithm
2. **Sarsa(λ)** - Temporal-difference learning with accumulating eligibility traces

The assignment is part of a Master's degree in Artificial Intelligence, focusing on tabular reinforcement learning methods.

## 🎮 Environment

- **Environment**: `TextFlappyBird-v0` (text-based Flappy Bird game)
- **Observation**: Discrete tuple `(dx, dy)` - distances from player to next pipe gap center
- **Action Space**: 2 actions (no-op / flap)
- **Reward**: +1 for each timestep survived, -1 for collision

## 🤖 Algorithms Implemented

### 1. Monte Carlo Control
- **Policy**: ε-greedy exploration-exploitation
- **Update Rule**: First-visit incremental mean of returns
- **Key Features**:
  - Model-free, off-policy learning
  - Learns from complete episodes
  - Epsilon decay for convergence

### 2. Sarsa(λ) with Accumulating Traces
- **Learning Rule**: Temporal-difference update with eligibility traces
- **Trace Type**: Accumulating (not replacing)
- **Formula**:
  - δₜ = Rₜ₊₁ + γQ(Sₜ₊₁, Aₜ₊₁) - Q(Sₜ, Aₜ)
  - e(Sₜ, Aₜ) ← e(Sₜ, Aₜ) + 1
  - Q(s,a) ← Q(s,a) + α·δₜ·e(s,a)
  - e(s,a) ← γ·λ·e(s,a)

## 📊 Analyses Included

1. **Learning Curves Comparison** - Side-by-side training performance
2. **Greedy Policy Evaluation** - Mean, std, and max returns over 100 episodes
3. **State-Value Function Visualization** - Heatmaps of learned V(s) = max_a Q(s,a)
4. **Parameter Sensitivity Analysis**:
   - λ (trace decay) for Sarsa(λ)
   - α (learning rate) for Sarsa(λ)
   - ε (exploration rate) for both agents
5. **Algorithm Comparison** - Convergence speed, final performance, states learned

## 🛠️ Setup & Installation

```bash
# Install Text Flappy Bird environment
pip install git+https://gitlab-research.centralesupelec.fr/stergios.christodoulidis/text-flappy-bird-gym.git

# Install dependencies
pip install gymnasium numpy matplotlib tqdm
```

## 🚀 Running the Notebook

1. Open `TFB_RL_Assignment.ipynb` in Jupyter
2. Run cells sequentially to:
   - Verify GPU/CPU setup
   - Create environment
   - Train Monte Carlo agent
   - Train Sarsa(λ) agent
   - Generate visualizations and analyses

## 📈 Key Results

- **Training Episodes**: 20,000 per agent
- **Performance Metric**: Episode return (sum of rewards)
- **Learning Window**: 200-episode moving average for smoother curves
- **Convergence**: Both agents converge to stable policies with different learning speeds

## 📁 Project Structure

```
Individual_assignment/
├── TFB_RL_Assignment.ipynb    # Main notebook with all implementations
├── README.md                  # This file
└── CONTEXTE.md               # Detailed project context
```

## 🎯 Hyperparameters

### Monte Carlo
- γ (discount): 1.0 (full episode rewards)
- ε (initial exploration): 0.1
- ε decay: 0.9999 per episode
- ε_min: 0.01

### Sarsa(λ)
- α (learning rate): 0.1
- γ (discount): 1.0
- λ (trace decay): 0.9
- ε (initial exploration): 0.1
- ε decay: 0.9999 per episode
- ε_min: 0.01

## 📚 References

- Sutton & Barto (2018): Reinforcement Learning: An Introduction
- Section 12.7: Sarsa(λ) with accumulating traces
- Text Flappy Bird Gym: `text-flappy-bird-gym` package

## 👤 Author

Master's Student - Artificial Intelligence (IA)
Reinforcement Learning Course Assignment

## 📝 Notes

- GPU acceleration checked but not required
- All computations can run on CPU
- Training time: ~5-10 minutes per agent on standard hardware
- Results are stochastic due to random initialization and exploration
