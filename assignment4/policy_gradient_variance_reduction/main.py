from envs.simple_env import SimpleEnv
from agents.standard_pg_agent import StandardPGAgent
from agents.rtg_pg_agent import RTGPGAgent
from agents.baseline_pg_agent import BaselinePGAgent

def train_agent(agent, episodes=500):
    env = SimpleEnv()
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_rewards = []
        states, actions, rewards = [], [], []

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            episode_rewards.append(reward)

        agent.learn(states, actions, rewards)
        total_rewards.append(sum(episode_rewards))

    env.close()
    return total_rewards

def main():
    standard_agent = StandardPGAgent(input_size=4, output_size=2)
    rtg_agent = RTGPGAgent(input_size=4, output_size=2)
    baseline_agent = BaselinePGAgent(input_size=4, output_size=2)

    standard_rewards = train_agent(standard_agent)
    rtg_rewards = train_agent(rtg_agent)
    baseline_rewards = train_agent(baseline_agent)

    # Here you can add code to calculate and compare the variances

if __name__ == "__main__":
    main()