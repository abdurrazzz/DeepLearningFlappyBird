import numpy as np
import pygame
import random
from collections import defaultdict, deque
from flappybird import *

class FlappyBirdAI:
    def __init__(self, learning_rate=0.2, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.9997):
        self.q_table = defaultdict(lambda: np.zeros(2))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.05
        self.frames_alive = 0
        self.best_score = 0
        self.consecutive_flaps = 0

    def discretize_state(self, bird_y, bird_vel, pipe_x, pipe_top_y, pipe_bottom_y):
        """Convert continuous state space to discrete state space with improved vertical control."""
        gap_center = (pipe_top_y + (pipe_bottom_y - pipe_top_y) / 2)
        distance_to_gap = bird_y - gap_center
        
        # More fine-grained vertical position bins
        vertical_position = int(np.clip(distance_to_gap / 30, -8, 8))
        
        # Include velocity in more detail
        velocity = int(np.clip(bird_vel / 2, -3, 3))
        
        # Distance to pipe with more detail when close
        horizontal_distance = int(np.clip(pipe_x / 100, 0, 5))
        
        # Add boundary awareness
        too_high = 1 if bird_y < 100 else 0
        too_low = 1 if bird_y > WIN_HEIGHT - 100 else 0
        
        return (vertical_position, horizontal_distance, velocity, too_high, too_low)

    def get_action(self, state):
        """Choose action using epsilon-greedy policy with improved movement control."""
        if random.random() < self.epsilon:
            if state[4] == 1:  # Too low
                self.consecutive_flaps = 0
                return 1  # Force flap
            elif state[3] == 1:  # Too high
                self.consecutive_flaps = 0
                return 0  # Force not flap
            elif self.consecutive_flaps >= 3:
                self.consecutive_flaps = 0
                return 0  # Force not flap after too many consecutive flaps
            else:
                action = random.randint(0, 1)
                if action == 1:
                    self.consecutive_flaps += 1
                else:
                    self.consecutive_flaps = 0
                return action
        
        action = np.argmax(self.q_table[state])
        if action == 1:
            self.consecutive_flaps += 1
        else:
            self.consecutive_flaps = 0
        return action

    def calculate_reward(self, bird, pipes, score, done):
        """Calculate reward with improved vertical positioning incentives."""
        self.frames_alive += 1
        
        if done:
            if bird.y <= 0 or bird.y >= WIN_HEIGHT:  # Hit ceiling or ground
                return -150
            return -100 if self.frames_alive < 100 else -50
        
        # Find nearest pipe
        nearest_pipe = None
        min_distance = float('inf')
        for pipe in pipes:
            if pipe.x + pipe.WIDTH >= bird.x:
                if pipe.x < min_distance:
                    min_distance = pipe.x
                    nearest_pipe = pipe
        
        if nearest_pipe is None:
            return 0.5
            
        # Calculate vertical distance to pipe gap center
        gap_center = (nearest_pipe.top_height_px + 
                     (WIN_HEIGHT - nearest_pipe.bottom_height_px)) / 2
        vertical_distance = abs(bird.y - gap_center)
        
        # Base reward
        reward = 0.5
        
        # Vertical positioning rewards
        if vertical_distance < 30:
            reward += 3
        elif vertical_distance < 60:
            reward += 1.5
        elif vertical_distance < 90:
            reward += 0.5
            
        # Boundary penalties
        if bird.y < 50 or bird.y > WIN_HEIGHT - 50:
            reward -= 2
            
        # Excessive flapping penalty
        if self.consecutive_flaps > 3:
            reward -= 1
            
        # Pipe passing reward
        if nearest_pipe.x + nearest_pipe.WIDTH < bird.x and not nearest_pipe.score_counted:
            reward += 25
            if score > self.best_score:
                self.best_score = score
                reward += 10
                
        return reward

    def learn(self, state, action, reward, next_state, done):
        """Update Q-values using Q-learning algorithm."""
        old_value = self.q_table[state][action]
        if done:
            next_max = 0
        else:
            next_max = np.max(self.q_table[next_state])
        
        new_value = (1 - self.learning_rate) * old_value + \
                   self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[state][action] = new_value
        
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

def train_ai_flappybird(num_episodes=1000):
    """Train the AI agent."""
    pygame.init()
    display_surface = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption('AI Flappy Bird')
    clock = pygame.time.Clock()
    images = load_images()
    
    agent = FlappyBirdAI()
    
    episode_rewards = []
    episode_scores = []
    best_score = 0
    
    for episode in range(num_episodes):
        bird = Bird(50, int(WIN_HEIGHT/2 - Bird.HEIGHT/2), 2,
                   (images['bird-wingup'], images['bird-wingdown']))
        pipes = deque()
        score = 0
        done = False
        episode_reward = 0
        agent.frames_alive = 0
        agent.consecutive_flaps = 0
        
        # Initialize first pipe
        pp = PipePair(images['pipe-end'], images['pipe-body'])
        pipes.append(pp)
        last_pipe_time = pygame.time.get_ticks()
        
        while not done:
            clock.tick(FPS)
            
            # Add new pipe with proper timing
            if pygame.time.get_ticks() - last_pipe_time >= PipePair.ADD_INTERVAL:
                if not pipes or pipes[-1].x < WIN_WIDTH - PipePair.WIDTH * 2.5:
                    pp = PipePair(images['pipe-end'], images['pipe-body'])
                    pipes.append(pp)
                    last_pipe_time = pygame.time.get_ticks()
            
            # Get current state
            nearest_pipe = None
            for pipe in pipes:
                if pipe.x + pipe.WIDTH >= bird.x:
                    nearest_pipe = pipe
                    break
            
            if nearest_pipe:
                state = agent.discretize_state(
                    bird.y,
                    bird.msec_to_climb,
                    nearest_pipe.x - bird.x,
                    nearest_pipe.top_height_px,
                    WIN_HEIGHT - nearest_pipe.bottom_height_px
                )
                
                # Choose and perform action
                action = agent.get_action(state)
                if action == 1:
                    bird.msec_to_climb = Bird.CLIMB_DURATION
            
                # Update game state
                for p in pipes:
                    p.update()
                bird.update()
                
                # Check for collisions
                pipe_collision = any(p.collides_with(bird) for p in pipes)
                if (pipe_collision or 
                    0 >= bird.y or 
                    bird.y >= WIN_HEIGHT - Bird.HEIGHT):
                    done = True
                
                # Update score
                for p in pipes:
                    if not p.score_counted and p.x + p.WIDTH < bird.x:
                        score += 1
                        p.score_counted = True
                
                # Calculate reward
                reward = agent.calculate_reward(bird, pipes, score, done)
                episode_reward += reward
                
                # Get next state and learn
                if nearest_pipe:
                    next_state = agent.discretize_state(
                        bird.y,
                        bird.msec_to_climb,
                        nearest_pipe.x - bird.x,
                        nearest_pipe.top_height_px,
                        WIN_HEIGHT - nearest_pipe.bottom_height_px
                    )
                    agent.learn(state, action, reward, next_state, done)
            
            # Remove invisible pipes
            while pipes and not pipes[0].visible:
                pipes.popleft()
            
            # Update display
            for x in (0, WIN_WIDTH / 2):
                display_surface.blit(images['background'], (x, 0))
            
            for p in pipes:
                display_surface.blit(p.image, p.rect)
            
            display_surface.blit(bird.image, bird.rect)
            
            # Display score and epsilon
            font = pygame.font.SysFont(None, 32, bold=True)
            score_surface = font.render(f'Score: {score} Best: {best_score}', True, (255, 255, 255))
            epsilon_surface = font.render(f'Epsilon: {agent.epsilon:.3f}', True, (255, 255, 255))
            flaps_surface = font.render(f'Consecutive Flaps: {agent.consecutive_flaps}', True, (255, 255, 255))
            display_surface.blit(score_surface, (10, 10))
            display_surface.blit(epsilon_surface, (10, 40))
            
            pygame.display.flip()
        
        if score > best_score:
            best_score = score
            
        episode_rewards.append(episode_reward)
        episode_scores.append(score)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_score = np.mean(episode_scores[-10:])
            print(f"Episode {episode + 1}")
            print(f"Average Reward (last 10): {avg_reward:.2f}")
            print(f"Average Score (last 10): {avg_score:.2f}")
            print(f"Best Score: {best_score}")
            print(f"Epsilon: {agent.epsilon:.3f}")
            print("------------------------")
    
    pygame.quit()
    return agent, episode_rewards, episode_scores

if __name__ == '__main__':
    agent, rewards, scores = train_ai_flappybird(num_episodes=1000)
    
    print("\nTraining Complete!")
    print(f"Final Average Reward (last 100): {np.mean(rewards[-100:]):.2f}")
    print(f"Final Average Score (last 100): {np.mean(scores[-100:]):.2f}")
    print(f"Best Score Achieved: {max(scores)}")