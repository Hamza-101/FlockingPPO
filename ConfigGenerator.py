import random
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from Settings import *
from tqdm import tqdm

TotalConfigs = 0

class Agent:
    def __init__(self):
        self.position = np.array([round(random.uniform(-SimulationVariables["X"], SimulationVariables["X"]), 1), round(random.uniform(SimulationVariables["Y"], SimulationVariables["Y"]), 1)])
        
    # Connectedness
    def get_neighbors(agent, agents, radius):
        neighbors = []
        
        for other_agent in agents:
            if agent != other_agent:
                distance = np.linalg.norm(agent.position - other_agent.position)
                if distance < radius:
                    neighbors.append(other_agent)
                    return True

        return False

    #Safety
    def check_min_distance(agent, agents, threshold):
        for other_agent in agents:
            if agent != other_agent:
                distance = np.linalg.norm(agent.position - other_agent.position)
                if distance < threshold:
                    return False

        return True

class Encoder(json.JSONEncoder):
    def default(self, obj):
        return json.JSONEncoder.default(self, obj)

def get_agent_locations(agents):
    agent_locations = [agent.position.tolist() for agent in agents]
    return agent_locations


def save_config(ConfigValidation, agent_locations):
    global TotalConfigs  

    # Create the "Directory" folder if it doesn't exist
    if not os.path.exists(Results["Directory"]):
        os.makedirs(Results["Directory"])

    while True:
        # Find the next available configuration number
        folder_name = f"{Results['Directory']}/Config_{TotalConfigs}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            break
        TotalConfigs += 1

    # Save the agent locations to a JSON file in the configuration folder
    filename = f"{folder_name}/config.json"
    with open(filename, "w") as f:
        json.dump(agent_locations, f, cls=Encoder)
        f.write("\n")

    return folder_name


def property_check(agents):
    # Dictionary to store neighbors for each agent
    agent_neighbors = {agent: [] for agent in agents}

    # Check safety and connectedness for each agent
    for agent in agents:
        # Safety distance check
        if not Agent.check_min_distance(agent, agents, SimulationVariables["SafetyRadius"]):
            return False

        # Check connectedness within the neighborhood distance
        for other_agent in agents:
            if agent != other_agent:
                distance = np.linalg.norm(agent.position - other_agent.position)
                if distance < SimulationVariables["NeighborhoodRadius"]:
                    agent_neighbors[agent].append(other_agent)

    # Check if there are no neighbors within the neighborhood distance for ally agent
    if any(not neighbors for neighbors in agent_neighbors.values()):
        return False

    # Return True if all agents pass both safety and connectedness checks
    return True

def plot_agents(agents, folder_name):

    x_coordinates = [agent.position[0] for agent in agents]
    y_coordinates = [agent.position[1] for agent in agents]

    plt.figure(figsize=(SimulationVariables["X"] + 5, SimulationVariables["Y"] + 5))
    plt.scatter(x_coordinates, y_coordinates, c = 'red', label = 'Agents', marker = 'o')

    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')

    PlotTitle = folder_name.replace('s','')
    plt.title(PlotTitle)

    # Set equal aspect ratio for axis scaling (1:1)
    plt.axis("equal")
    
    plt.xlim(-(SimulationVariables["X"]), SimulationVariables["X"])
    plt.ylim(-(SimulationVariables["Y"]), SimulationVariables["Y"])

    plt.legend()
    plt.grid(True)
    plt.show()

j = 0


for i in tqdm(range(1, 1000)):
    agents = [Agent() for _ in range(SimulationVariables["SimAgents"])]

    check_result = property_check(agents)
    AgentsData = get_agent_locations(agents)
    if check_result == True:
        folder_name = save_config(check_result, AgentsData)
        j = j + 1
print(j)
