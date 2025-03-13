import os
import json
from tqdm import tqdm
from Settings import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

fig, ax = plt.subplots()

def read_data(file):
    with open(file, "r") as f:
        all_data = json.load(f)
    agent_data = {}

    for agent_id, agent_points in all_data.items():
        agent_data[int(agent_id)] = agent_points

    max_num_points = max(len(agent_points) for agent_points in agent_data.values())

    return agent_data, max_num_points

def function(i, agent_data, acc_data, vel_data, traj_data, distances_data):
    plt.clf()
    plt.cla()

    plt.axis('equal')  

    # Plot agents, arrows, and calculate distances
    for agent_id, data_points in agent_data.items():
        if i < len(data_points):
            data_point = data_points[i]
            x_pos = data_point[0] 
            y_pos = data_point[1] 

            plt.scatter(x_pos, y_pos, c='r', marker='o', s=20)

            acc_x, acc_y = acc_data[agent_id][i]
            acc_x_scaled, acc_y_scaled = acc_x * 0.1, acc_y * 0.1
            plt.arrow(x_pos, y_pos, acc_x_scaled, acc_y_scaled, color='blue', width=0.01, head_width=0.05)

            vel_x, vel_y = vel_data[agent_id][i]
            vel_x_scaled, vel_y_scaled = vel_x * 0.1, vel_y * 0.1
            plt.arrow(x_pos, y_pos, vel_x_scaled, vel_y_scaled, color='green', width=0.01, head_width=0.05)

    # Draw edges if agents come closer than the neighborhood radius of 10
    for agent_id, data_points in agent_data.items():
        for other_agent_id, other_data_points in agent_data.items():
            if agent_id != other_agent_id and i < len(data_points) and i < len(other_data_points):
                distance = ((data_points[i][0] - other_data_points[i][0])**2 + (data_points[i][1] - other_data_points[i][1])**2)**0.5
                if distance < 2:
                    x1, y1 = data_points[i]
                    x2, y2 = other_data_points[i]
                    plt.plot([x1, x2], [y1, y2], color='black', linewidth=0.5)

    # Get distances for the current timestep
    if i < len(distances_data):
        distances = distances_data[i]
        legend_entries_distances = [
            Line2D([0], [0], color='white', lw=2, label=f"D1 (A0-A1): {distances['0'][0]:.2f}"),
            Line2D([0], [0], color='white', lw=2, label=f"D2 (A1-A2): {distances['1'][1]:.2f}"),
            Line2D([0], [0], color='white', lw=2, label=f"D3 (A2-A0): {distances['2'][0]:.2f}")
        ]
    else:
        legend_entries_distances = []

    # Create the legend
    legend_entries = [
        Line2D([0], [0], color='red', marker='o', label="Agent Position", lw=0),
        Line2D([0], [0], color='blue', lw=2, label="Acceleration"),
        Line2D([0], [0], color='green', lw=2, label="Velocity"),
        *legend_entries_distances
    ]

    # Display legend
    plt.legend(handles=legend_entries, loc='upper right')

    plt.grid(True)

def AnimateEpisode():
    for j in range(0, SimulationVariables["Episodes"]):
        print(j)
        # Read position, acceleration, velocity, trajectory, and distances
        file = rf"Results\Flocking\Testing\Episodes\Episode{j}_positions.json"
        agent_data, max_num_points = read_data(file)

        acc_file = rf"Results\Flocking\Testing\Episodes\Episode{j}_accelerations.json"
        acc_data, _ = read_data(acc_file)

        vel_file = rf"Results\Flocking\Testing\Episodes\Episode{j}_velocities.json"
        vel_data, _ = read_data(vel_file)

        traj_file = rf"Results\Flocking\Testing\Episodes\Episode{j}_trajectory.json"
        traj_data, _ = read_data(traj_file)

        dist_file = rf"Results\Flocking\Testing\Episodes\Episode{j}_distances.json"
        with open(dist_file, "r") as f:
            distances_data = json.load(f)

        for i in tqdm(range(SimulationVariables["EvalTimeSteps"])):
            function(i, agent_data, acc_data, vel_data, traj_data, distances_data)

        anim = animation.FuncAnimation(
            fig, function,
            fargs=(agent_data, acc_data, vel_data, traj_data, distances_data),
            frames=Animation["TimeSteps"], interval=100, blit=False
        )
        anim.save(rf"Results\Flocking\Testing\Episodes\Episode{j}.mp4", writer="ffmpeg")

if __name__ == "__main__":
    AnimateEpisode()
