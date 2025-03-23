# Flappy Bird Q-Learning AI

A reinforcement learning implementation for the classic Flappy Bird game using Q-learning algorithm, with a Streamlit dashboard for visualizing training metrics.

## Overview

This project implements an AI agent that learns to play Flappy Bird through Q-learning, a model-free reinforcement learning algorithm. The agent improves its gameplay by exploring the environment, receiving rewards for good actions, and learning optimal strategies over time.

## Features

- **Q-learning Implementation**: The AI agent uses a state discretization approach to learn optimal actions
- **Adaptive Exploration**: Implements epsilon-greedy exploration with decay to balance exploration and exploitation
- **Performance Metrics**: Tracks various metrics including score, rewards, and survival time
- **Training Visualization**: Interactive Streamlit dashboard to visualize the agent's learning progress

## Project Structure

```
flappybird-qlearning/
├── flappybird.py         # Base Flappy Bird game implementation (required but not included)
├── flappybird_ai.py      # Q-learning agent implementation
├── training_dashboard.py # Streamlit visualization dashboard
├── requirements.txt      # Project dependencies
└── README.md             # This README file
```

## Q-learning Implementation

The agent uses the following reinforcement learning components:

- **State Space**: Discretized representation of bird position, velocity, pipe distance and gap position
- **Action Space**: Two possible actions (flap or don't flap)
- **Reward System**: Rewards for staying alive, positioning near pipe gaps, and penalties for collisions
- **Learning Algorithm**: Q-learning with decaying epsilon-greedy exploration policy

The agent improves its policy by updating Q-values according to the equation:

```
Q(s,a) = (1-α) * Q(s,a) + α * (r + γ * max(Q(s',a')))
```

Where:
- α (alpha) is the learning rate
- γ (gamma) is the discount factor
- s is the current state
- a is the action taken
- r is the reward received
- s' is the next state

## Visualization Dashboard

The Streamlit dashboard provides:

- **Performance Metrics**: Charts for game score, survival time, and overall performance
- **Learning Progress**: Visualization of rewards and exploration rate over training episodes
- **Detailed Analysis**: Statistical insights and correlations between different metrics

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Pygame
- Numpy
- Pandas
- Streamlit
- Plotly

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/flappybird-qlearning.git
   cd flappybird-qlearning
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure you have the base `flappybird.py` file in the same directory (contains the game mechanics)

### Running the AI Training

To train the AI agent:

```
python flappybird_ai.py
```

This will start the training process. The agent will play multiple episodes, learning from each attempt. Training metrics will be displayed in the console.

### Visualizing Training Results

To launch the visualization dashboard:

```
streamlit run training_dashboard.py
```

This will start a local server and open the dashboard in your web browser. From there, you can:
- Upload training metrics JSON files
- View performance charts and statistics
- Analyze the agent's learning progress

## Customization

You can adjust the agent's learning parameters in `flappybird_ai.py`:

- `learning_rate`: Controls how quickly the agent updates its Q-values
- `discount_factor`: Determines the importance of future rewards
- `epsilon`: Initial exploration rate
- `epsilon_decay`: Rate at which exploration decreases over time

## How It Works

1. **State Discretization**: The continuous game state is converted to a discrete representation:
   - Vertical distance to pipe gap center
   - Horizontal distance to next pipe
   - Bird velocity
   - Boundary awareness (too high/low flags)

2. **Reward System**: The agent receives rewards based on:
   - Survival time (+0.5 per frame)
   - Positioning near pipe gaps (up to +3)
   - Passing pipes successfully (+25)
   - Penalties for hitting boundaries (-150)
   - Penalties for excessive flapping (-1)

3. **Action Selection**: Uses epsilon-greedy policy with safety constraints:
   - Forces flapping when too low
   - Prevents flapping when too high
   - Limits consecutive flaps

4. **Learning**: Updates Q-values after each action based on observed rewards

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project is inspired by the classic Flappy Bird game
- The reinforcement learning implementation is based on Q-learning principles
