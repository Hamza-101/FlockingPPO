import os
import gym
import json
import numpy as np
import torch as th
from tqdm import tqdm
from Settings import *
from gym import spaces
import matplotlib
matplotlib.use('Agg')  # faster than rendering plots
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, get_system_info
from stable_baselines3.common.callbacks import BaseCallback
from scipy.signal import savgol_filter
import shutil
from PlotAnimationRL import *
import glob
from scipy.spatial import cKDTree
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

positions_directory = "Results/Flocking/Testing/Episodes"  

policy_kwargs = dict(
    activation_fn=th.nn.Tanh,  
    net_arch=[dict(pi=[512, 512], vf=[512, 512])]   # decreased from 7 layers
)

class Encoder(json.JSONEncoder):
    def default(self, obj):
        return json.JSONEncoder.default(self, obj)  

class TQDMProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super(TQDMProgressCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self) -> bool:
        if self.pbar:
            self.pbar.update(self.model.num_timesteps - self.pbar.n)
        return True

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()
  
class Agent:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        self.acceleration = np.zeros(2)
        self.max_acceleration = SimulationVariables["AccelerationUpperLimit"]
        self.velocity = np.round(np.random.uniform(-SimulationVariables["VelocityUpperLimit"], SimulationVariables["VelocityUpperLimit"], size=2), 2)       
        self.max_velocity = SimulationVariables["VelocityUpperLimit"]


    def update(self, action):
        self.acceleration += action
        acc_magnitude = np.linalg.norm(self.acceleration)
        if acc_magnitude > 0:  
            if acc_magnitude > SimulationVariables["AccelerationUpperLimit"]:
                scaled_magnitude = SimulationVariables["AccelerationUpperLimit"] * np.tanh(acc_magnitude / SimulationVariables["AccelerationUpperLimit"])
                self.acceleration = (self.acceleration / acc_magnitude) * scaled_magnitude
        self.velocity += self.acceleration * SimulationVariables["dt"]
        vel = np.linalg.norm(self.velocity)
        if vel > 0:
            if vel > self.max_velocity:
                self.velocity = self.velocity * np.tanh(self.max_velocity / vel)

        self.position += self.velocity * SimulationVariables["dt"]

        return self.position, self.velocity

class FlockingEnv(gym.Env):
    def __init__(self, seed=None):
        super(FlockingEnv, self).__init__()
        self.seed(seed)
        self.episode=0
        self.counter=0
        self.CTDE=False
        self.current_timestep = 0
        self.reward_log = []
        self.np_random, _ = gym.utils.seeding.np_random(None)
        self.cumulative_rewards = {i: 0 for i in range(SimulationVariables["SimAgents"])}
        
        # reward buffer for CTDE, write to disk at end of episode
        self.alignment_rewards_buffer = []
        self.cohesion_rewards_buffer = []
        
        self.agents = [Agent(position) for position in self.read_agent_locations()]

        min_action = np.array([-5, -5] * len(self.agents), dtype=np.float32)
        max_action = np.array([5, 5] * len(self.agents), dtype=np.float32)
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)

        min_obs = np.array([-np.inf, -np.inf, -2.5, -2.5] * len(self.agents), dtype=np.float32)
        max_obs = np.array([np.inf, np.inf, 2.5, 2.5] * len(self.agents), dtype=np.float32)
        self.observation_space = spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]
        
    def step(self, actions):
        training_rewards = {}
        # noise = NormalActionNoise(mean=np.zeros(len(actions)), sigma=0.5 * np.ones(len(actions)))
        # noisy_actions = actions + noise
        self.current_timestep += 1
        reward=0
        done=False
        info={}
        observations = self.simulate_agents(actions)
        reward, out_of_flock = self.calculate_reward() 

        if (self.CTDE==False): # no out of flock
            for agent in self.agents:
                if (out_of_flock==True): 
                    done=True
                    env.reset()

        return observations, reward, done, info

    def reset(self):   
        env.seed(SimulationVariables["Seed"])
        self.agents = [Agent(position) for position in self.read_agent_locations()]
        for agent in self.agents:
            agent.acceleration = np.zeros(2)
            agent.velocity = np.round(np.random.uniform(-SimulationVariables["VelocityUpperLimit"], SimulationVariables["VelocityUpperLimit"], size=2), 2)             

        observation = self.get_observation().flatten()
        self.current_timestep = 0  
        return observation   

    def close(self):
        print("Simulation is complete. Cleaned Up!.")
        
    def simulate_agents(self, actions):
        observations = []  
        actions_reshaped = actions.reshape(((SimulationVariables["SimAgents"]), 2))
        for i, agent in enumerate(self.agents):
            position, velocity = agent.update(actions_reshaped[i])
            observation_pair = np.concatenate([position, velocity])
            observations = np.concatenate([observations, observation_pair])
            
        return observations
    
    def check_collision(self, agent):
        for other in self.agents:
            if agent != other:
                distance = np.linalg.norm(agent.position - other.position)
                if distance < SimulationVariables["SafetyRadius"]:
                    return True  
        return False

    def get_observation(self):
        n_agents = len(self.agents)
        obs = np.zeros((n_agents, 4), dtype=np.float32)

        for i, agent in enumerate(self.agents):
                agent.position[0],
                agent.position[1],
                agent.velocity[0],
                agent.velocity[1]

        return obs
   
    def get_closest_neighbors(self, agent):
        neighbor_positions=[]
        neighbor_velocities=[]
                
        for _, other in enumerate(self.agents):
            if agent != other:
                distance = np.linalg.norm(other.position - agent.position)

                if(distance <= SimulationVariables["NeighborhoodRadius"]):
                        neighbor_positions.append(other.position)
                        neighbor_velocities.append(other.velocity)                        

        return neighbor_positions, neighbor_velocities         
   
    def calculate_reward(self):
        total_reward = 0
        out_of_flock = False
        cumulative_alignment = 0
        cumulative_cohesion = 0

        for i, agent in enumerate(self.agents):
            neighbor_positions, neighbor_velocities = self.get_closest_neighbors(agent)
            agent_reward, alignment_reward, cohesion_reward, out_of_flock = self.reward(agent, neighbor_velocities, neighbor_positions)
            self.cumulative_rewards[i] += agent_reward
            cumulative_alignment += alignment_reward
            cumulative_cohesion += cohesion_reward

            total_reward += agent_reward
            
        if(self.CTDE==True):
            # memory buffer
            self.alignment_rewards_buffer.append(cumulative_alignment)
            self.cohesion_rewards_buffer.append(cumulative_cohesion)

        return total_reward, out_of_flock

    # flush buffer to disk at end of ep
    def save_reward_buffers(self):
        if self.CTDE and self.alignment_rewards_buffer and self.cohesion_rewards_buffer:
            os.makedirs(positions_directory, exist_ok=True)
            
            with open(os.path.join(positions_directory, f"CohesionRewardsEpisode{self.episode}.json"), "w") as f:
                for reward in self.cohesion_rewards_buffer:
                    f.write(f"{reward}\n")
                    
            with open(os.path.join(positions_directory, f"AlignmentRewardsEpisode{self.episode}.json"), "w") as f:
                for reward in self.alignment_rewards_buffer:
                    f.write(f"{reward}\n")
                    
            # clear buffer
            self.alignment_rewards_buffer = []
            self.cohesion_rewards_buffer = []

    def reward(self, agent, neighbor_velocities, neighbor_positions):
        CohesionReward = 0
        AlignmentReward = 0
        SeparationReward = 0
        total_reward = 0
        outofflock = False

        c_weight = 1.3  # 1.0  
        a_weight = 0.2  # 0.3
        s_weight = 0.1  # 0.5

        if len(neighbor_positions) > 0:
            # cohesion reward -> minimise distance to centre
            group_center = np.mean(neighbor_positions, axis=0)
            d_to_center = np.linalg.norm(agent.position - group_center)
            if d_to_center <= SimulationVariables["SafetyRadius"]:
                CohesionReward = 15
            elif d_to_center <= SimulationVariables["NeighborhoodRadius"]:
                CohesionReward = 10 * (1 - (d_to_center - SimulationVariables["SafetyRadius"]) / (SimulationVariables["NeighborhoodRadius"] - SimulationVariables["SafetyRadius"]))
            else:
                CohesionReward = -20

            # alignment reward -> minimise angle and speed difference
            avg_velocity = np.mean(neighbor_velocities, axis=0)
            cos_angle = np.dot(agent.velocity, avg_velocity) / (np.linalg.norm(agent.velocity) * np.linalg.norm(avg_velocity) + 1e-6)
            angle_diff = np.arccos(np.clip(cos_angle, -1, 1))
            speed_diff = np.abs(np.linalg.norm(agent.velocity) - np.linalg.norm(avg_velocity))
            AlignmentReward = -2 * angle_diff - 0.5 * speed_diff        # penalise angle and speed difference

            # separation -> avoid collapse to centre of mass
            for neighbor_pos in neighbor_positions:
                distance = np.linalg.norm(agent.position - neighbor_pos)
                SeparationReward -= 10 * (1 - distance / SimulationVariables["SafetyRadius"])# **2 if necessary
                
        else:
            CohesionReward = -10    # penalty for no neighbors
            outofflock = True

        total_reward = c_weight * CohesionReward + a_weight * AlignmentReward + s_weight * SeparationReward

        return total_reward, AlignmentReward, CohesionReward, outofflock

    def read_agent_locations(self):
        File = os.path.join(Results["InitPositions"] + str(self.counter), "config.json")
        with open(File, "r") as f:
            data = json.load(f)
        return data

    def seed(self, seed=SimulationVariables["Seed"]):
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]

def delete_files(): 
    Paths = [
        "Results/Flocking/Testing/Dynamics/Accelerations",
        "Results/Flocking/Testing/Dynamics/Velocities",
        "Results/Flocking/Testing/Rewards/Other"
    ]

    Logs = [
        "AlignmentReward_log.json", "CohesionReward_log.json",
        "SeparationReward_log.json", "CollisionReward_log.json",
        "Reward_Total_log.json"
    ]

    for Path in Paths:
        for episode in range(0, 10):
            file_path = os.path.join(Files['Flocking'], Path, f"Episode{episode}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File {file_path} has been deleted.")

    for log_file in Logs:
        for episode in range(0, 10):
            file_path = os.path.join(Files['Flocking'], "Testing", "Rewards", "Components", f"Episode{episode}", log_file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File {file_path} has been deleted.")

    print("All specified files have been deleted.")

def setup_episode_folder(episode_name):
    episode_folder = os.path.join(positions_directory, episode_name)
    if os.path.exists(episode_folder):
        for file in os.listdir(episode_folder):
            os.remove(os.path.join(episode_folder, file)) 
    else:
        os.makedirs(episode_folder, exist_ok=True)
    return episode_folder

positions_directory = "Results/Flocking/Testing/Episodes" 

def generate_combined():
    directory = "Results/Flocking/Testing/Episodes"
    os.makedirs(directory, exist_ok=True)  

    fig_combined, ax_combined = plt.subplots(figsize=(12, 8))
    ax_combined_seconds = ax_combined.twiny()  

    cohesion_files = sorted([f for f in os.listdir(directory) if f.startswith("CohesionRewardsEpisode")])
    alignment_files = sorted([f for f in os.listdir(directory) if f.startswith("AlignmentRewardsEpisode")])

    episodes = sorted(set(
        f.split("CohesionRewardsEpisode")[1].split(".json")[0] for f in cohesion_files
    ).intersection(
        f.split("AlignmentRewardsEpisode")[1].split(".json")[0] for f in alignment_files
    ))

    if not episodes:
        print("No valid episodes found!")
        return

    for episode in episodes:
        cohesion_file = os.path.join(directory, f"CohesionRewardsEpisode{episode}.json")
        alignment_file = os.path.join(directory, f"AlignmentRewardsEpisode{episode}.json")

        if not os.path.exists(cohesion_file) or not os.path.exists(alignment_file):
            print(f"Skipping missing files for Episode {episode}")
            continue

        with open(cohesion_file, "r") as f:
            cohesion_rewards = [float(line.strip()) for line in f.readlines()][:200]
        with open(alignment_file, "r") as f:
            alignment_rewards = [float(line.strip()) for line in f.readlines()][:200]

        combined_rewards = [c + a for c, a in zip(cohesion_rewards, alignment_rewards)]

        timesteps = range(1, len(combined_rewards) + 1)
        seconds = [timestep / 10 for timestep in timesteps]  

        fig, ax = plt.subplots(figsize=(10, 6))
        ax_seconds = ax.twiny()  

        ax.plot(timesteps, cohesion_rewards, label="Cohesion Rewards", alpha=0.7, linestyle='--')
        ax.plot(timesteps, alignment_rewards, label="Alignment Rewards", alpha=0.7, linestyle='-.')
        ax.plot(timesteps, combined_rewards, label="Combined Rewards", alpha=0.7)

        ax.set_title(f"Rewards for Episode {episode}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Reward")
        ax.legend()
        ax.grid(True)

        ax.set_xlim(1, 200)
        ax_seconds.set_xlim(ax.get_xlim())
        ax_seconds.set_xticks(ax.get_xticks())
        ax_seconds.set_xticklabels([f"{tick / 10:.1f}" for tick in ax.get_xticks()])
        ax_seconds.set_xlabel("Time (seconds)")

        plt.tight_layout()
        plt.savefig(os.path.join(positions_directory, f"Episode_{episode}_Rewards.png"), dpi=300)
        plt.close(fig)

        ax_combined.plot(timesteps, combined_rewards, label=f"Combined Rewards (Episode {episode})", alpha=0.5)

    ax_combined.set_title("Combined Rewards - All Episodes (200 Timesteps)")
    ax_combined.set_xlabel("Timestep")
    ax_combined.set_ylabel("Reward")
    ax_combined.legend()
    ax_combined.grid(True)

    ax_combined.set_xlim(1, 200)
    ax_combined_seconds.set_xlim(ax_combined.get_xlim())
    ax_combined_seconds.set_xticks(ax_combined.get_xticks())
    ax_combined_seconds.set_xticklabels([f"{tick / 10:.1f}" for tick in ax_combined.get_xticks()])
    ax_combined_seconds.set_xlabel("Time (seconds)")

    plt.tight_layout()
    combined_plot_path = os.path.join(positions_directory, "Combined_Cohesion_Alignment_Rewards.png")
    plt.savefig(combined_plot_path, dpi=300)
    plt.close(fig_combined)
    
def generateVelocity(episode, episode_folder):
    velocities_dict = {}
    velocity_file_path = os.path.join(positions_directory, f"Episode{episode}_velocities.json")
    
    if not os.path.exists(velocity_file_path):
        print(f"File {velocity_file_path} not found.")
        return

    with open(velocity_file_path, 'r') as f:
        episode_velocities = json.load(f)

    for agent_id in range(6):  
        velocities_dict.setdefault(agent_id, []).extend(episode_velocities.get(str(agent_id), []))

    colors = plt.cm.get_cmap('tab10', 6)
    downsample_factor = 10  

    plt.figure(figsize=(12, 6))
    plt.clf()
    
    for agent_id in range(6):  
        agent_velocities = np.array(velocities_dict[agent_id])
        agent_velocities = savgol_filter(agent_velocities, window_length=3, polyorder=2, axis=0)  
        velocities_magnitude = np.sqrt(agent_velocities[:, 0]**2 + agent_velocities[:, 1]**2)
        
        plt.plot(velocities_magnitude[::downsample_factor], label=f"Agent {agent_id+1}", color=colors(agent_id), linewidth=1)

    plt.title(f"Velocity - Episode {episode}")
    plt.xlabel("Time Step")
    plt.ylabel("Velocity Magnitude")
    plt.ylim([0, 5])  
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(episode_folder, f"Episode_{episode}_Velocity.png"))
    plt.close()  
    print(f"Velocity plot saved for Episode {episode}")

def generateAcceleration(episode, episode_folder):
    acceleration_file_path = os.path.join(positions_directory, f"Episode{episode}_accelerations.json")
    
    if not os.path.exists(acceleration_file_path):
        print(f"File {acceleration_file_path} not found.")
        return

    with open(acceleration_file_path, 'r') as f:
        episode_accelerations = json.load(f)

    colors = plt.cm.get_cmap('tab10', 6)
    downsample_factor = 10  

    plt.figure(figsize=(12, 6))
    plt.clf()

    for agent_id in range(6):
        agent_accelerations = np.array(episode_accelerations[str(agent_id)])
        smoothed_accelerations = np.sqrt(agent_accelerations[:, 0]**2 + agent_accelerations[:, 1]**2)
        smoothed_accelerations = savgol_filter(smoothed_accelerations, window_length=15, polyorder=3, axis=0)  

        plt.plot(smoothed_accelerations[::downsample_factor], label=f"Agent {agent_id+1}", color=colors(agent_id), linewidth=1)

    plt.title(f"Acceleration - Episode {episode}")
    plt.xlabel("Time Step")
    plt.ylabel("Acceleration Magnitude")
    plt.ylim([0, 10])  
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(episode_folder, f"Episode_{episode}_Acceleration.png"))
    plt.close()  
    print(f"Acceleration plot saved for Episode {episode}")

def generatePlots():
    for episode in range(SimulationVariables["Episodes"]):
        episode_name = f"Episode{episode}".split('_')[0]
        episode_folder = setup_episode_folder(episode_name)
        
        generateVelocity(episode, episode_folder)
        generateAcceleration(episode, episode_folder)

def delete_existing_files(directory, pattern):
    files = glob.glob(os.path.join(directory, pattern))
    for file in files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")
#------------------------
class LossCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LossCallback, self).__init__(verbose)
        self.loss_threshold = 2000

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) >= 1000:
            recent_losses = [ep_info['loss'] for ep_info in self.model.ep_info_buffer[-1000:]]
            average_loss = np.mean(recent_losses)

            if average_loss < self.loss_threshold:
                print(f"Stopping training because average loss ({average_loss}) is below threshold.")
                return False  

        return True

class AdaptiveExplorationCallback(BaseCallback):
    def __init__(self, initial_ent_coef=0.40, min_ent_coef=1e-10, decay_rate=0.85, max_reward_threshold=60, verbose=0):
        super(AdaptiveExplorationCallback, self).__init__(verbose)
        self.initial_ent_coef = initial_ent_coef       
        self.min_ent_coef = min_ent_coef               
        self.decay_rate = decay_rate                   
        self.ent_coef = initial_ent_coef               
        self.max_reward_threshold = max_reward_threshold  

    def _on_training_start(self):
        self.model.ent_coef = self.initial_ent_coef

    def _on_step(self) -> bool:
        # cumulative rewards
        all_cumulative_rewards = self.model.env.get_attr('cumulative_rewards')
        
        # check env for any cumulative rewards above threshold
        any_env_above = any(
            all(reward >= self.max_reward_threshold for reward in env_rewards.values())
            for env_rewards in all_cumulative_rewards
        )
        
        if any_env_above:
            self.ent_coef = max(self.ent_coef * self.decay_rate, self.min_ent_coef)
        else:
            self.ent_coef = self.initial_ent_coef
        self.model.ent_coef = self.ent_coef
        return True
#------------------------
if __name__ == "__main__":
    if os.path.exists(Results["Rewards"]):
        os.remove(Results["Rewards"])
        print(f"File {Results['Rewards']} has been deleted.")

    if os.path.exists("training_rewards.json"):
        os.remove("training_rewards.json")
        print(f"File training_rewards has been deleted.")    

    def seed_everything(seed):
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        th.manual_seed(seed)
        th.cuda.manual_seed(seed)
        th.backends.cudnn.deterministic = True
        env.seed(seed)
        env.action_space.seed(seed)

    env = FlockingEnv()
    seed_everything(SimulationVariables["Seed"])

    loss_callback = LossCallback()
    adaptive_exploration_callback = AdaptiveExplorationCallback()
    progress_callback = TQDMProgressCallback(total_timesteps=SimulationVariables["LearningTimeSteps"])

    device = "cuda" if th.cuda.is_available() else "cpu"
    device = 'cpu'
    print(f"Using device: {device}")

    if device == "cpu":
        th.set_num_threads(10)

    policy = "MlpPolicy"
    ModelName = f"{Files['Flocking']}/Models/Flocking_{policy}_{SimulationVariables['LearningTimeSteps']}"

    
    num_cpu = 6  # parallel environments
    def make_env():
        env = FlockingEnv()
        env.CTDE = False  # disable cdte during training
        return env
        
    env = SubprocVecEnv([make_env for _ in range(num_cpu)])

    model = PPO(policy, env, device=device, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_flocking_tensorboard/", verbose=1)
    model.set_random_seed(SimulationVariables["ModelSeed"])

    model.learn(total_timesteps=SimulationVariables["LearningTimeSteps"],  
               callback=[progress_callback, adaptive_exploration_callback])

    model.save(f"{Files['Flocking']}/Models/Flocking_{policy}_Simulation")
    env = FlockingEnv()
    model = PPO.load(f"{Files['Flocking']}/Models/Flocking_{policy}_Simulation", device=device)

    delete_files()
    positions_directory = f"{Files['Flocking']}/Testing/Episodes/"
    os.makedirs(f"{Files['Flocking']}/Models", exist_ok=True)
    os.makedirs(positions_directory, exist_ok=True)

    env.counter=0
    episode_rewards_dict = {}
    positions_dict = {i: [] for i in range(len(env.agents))}

    delete_existing_files(positions_directory, "CohesionRewardsEpisode*.json")
    delete_existing_files(positions_directory, "AlignmentRewardsEpisode*.json")

    for episode in tqdm(range(0, SimulationVariables["Episodes"])):
        env.episode = episode
        obs = env.reset()
        env.CTDE = True  # enable ctde during eval
        done = False
        timestep = 0
        reward_episode = []

        distances_dict = []
        positions_dict = {i: [] for i in range(len(env.agents))}
        velocities_dict = {i: [] for i in range(len(env.agents))}
        accelerations_dict = {i: [] for i in range(len(env.agents))}
        trajectory_dict = {i: [] for i in range(len(env.agents))}
        
        print(f"\n--- Episode {episode} ---")  
        print(env.counter)

        for i, agent in enumerate(env.agents):
            accelerations_dict[i].append(agent.acceleration.tolist())
            velocities_dict[i].append(agent.velocity.tolist())
            positions_dict[i].append(agent.position.tolist())
            trajectory_dict[i].append(agent.position.tolist())

        while timestep < SimulationVariables["EvalTimeSteps"]:
            actions, state = model.predict(obs)
            obs, reward, done, info = env.step(actions)
            reward_episode.append(reward)
            
            timestep_distances = {}  
            
            for i, agent in enumerate(env.agents):
                positions_dict[i].append(agent.position.tolist())
                velocity = agent.velocity.tolist()
                velocities_dict[i].append(velocity)
                acceleration = agent.acceleration.tolist()
                accelerations_dict[i].append(acceleration)
                trajectory_dict[i].append(agent.position.tolist())
                
                distances = []
                for j, other_agent in enumerate(env.agents):
                    if i != j:  
                        distance = np.linalg.norm(np.array(other_agent.position) - np.array(agent.position))
                        distances.append(distance)
                timestep_distances[i] = distances

            distances_dict.append(timestep_distances)

            timestep += 1
            episode_rewards_dict[str(episode)] = reward_episode
            
        # write reward buffer at end of ep
        env.save_reward_buffers()

        with open(os.path.join(positions_directory, f"Episode{episode}_positions.json"), 'w') as f:
            json.dump(positions_dict, f, indent=4)
        with open(os.path.join(positions_directory, f"Episode{episode}_velocities.json"), 'w') as f:
            json.dump(velocities_dict, f, indent=4)
        with open(os.path.join(positions_directory, f"Episode{episode}_accelerations.json"), 'w') as f:
            json.dump(accelerations_dict, f, indent=4)
        with open(os.path.join(positions_directory, f"Episode{episode}_distances.json"), 'w') as f:
            json.dump(distances_dict, f, indent=4)  
        with open(os.path.join(positions_directory, f"Episode{episode}_trajectory.json"), 'w') as f:
            json.dump(trajectory_dict, f, indent=4)

        env.counter += 1
        print(sum(reward_episode))

    with open(rf"{Results['EpisodalRewards']}.json", 'w') as f:
        json.dump(episode_rewards_dict, f, indent=4)

    env.close()

    generatePlots()
    generate_combined()
