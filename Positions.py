import os
import matplotlib.pyplot as plt
import json

output_path = r"Results\Flocking\Testing\Episodes"

def plot_trajectories():
    for i in range(0, 10):    
        file_path = os.path.join(output_path, f"Episode{i}_positions.json")
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue  # Skip if the file doesn't exist

        with open(file_path, "r") as file:
            data = json.load(file)

        plt.figure(figsize=(10, 10))
        for agent_id, positions in data.items():
            positions = list(map(tuple, positions))  # Convert lists to tuples for plotting
            x, y = zip(*positions)  # Unpack x and y coordinates

            # Plot trajectory as a line
            plt.plot(x, y, linestyle="-", label=f"Agent {agent_id}")

            # Mark final position with a dot
            plt.scatter(x[-1], y[-1], s=100, marker="o", label=f"Final {agent_id}")

        # Configure plot
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title(f"Agent Trajectories - Episode {i}")
        plt.grid(True)

        # Save the figure
        save_path = os.path.join(output_path, f"Episode{i}_trajectories.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")

plot_trajectories()


#Normal Version

# import os
# import matplotlib.pyplot as plt
# import json

# output_path = r"Test3\Results\Flocking\Testing\Episodes"

# def plot_trajectories():
#     for i in range(0, 10):    
#         file_path = os.path.join(output_path, f"Episode{i}_positions.json")
        
#         if not os.path.exists(file_path):
#             print(f"File not found: {file_path}")
#             continue  # Skip if the file doesn't exist

#         with open(file_path, "r") as file:
#             data = json.load(file)

#         plt.figure(figsize=(10, 10))
#         for agent_id, positions in data.items():
#             positions = list(map(tuple, positions))  # Convert lists to tuples for plotting
#             x, y = zip(*positions)

#             # Plot trajectory as a black line
#             plt.plot(x, y, linestyle="-", color="black")

#             # Mark final position with a filled light yellow circle
#             plt.scatter(x[-1], y[-1], s=100, color="yellow", edgecolors="black", marker="o")

#         # Configure plot
#         plt.xlabel("X Position")
#         plt.ylabel("Y Position")
#         plt.title(f"Agent Trajectories - Episode {i}")
#         plt.grid(True)

#         # Save the figure
#         save_path = os.path.join(output_path, f"Episode{i}_trajectories.png")
#         plt.savefig(save_path, dpi=300, bbox_inches="tight")
#         plt.close()

#         print(f"Saved: {save_path}")

# plot_trajectories()
