import subprocess

# 1) Run the first script and wait for it to finish
subprocess.run(["python", "PPO.py"])

# 2) After the first script completes, run the second
subprocess.run(["python", "PlotAnimationRL.py"])

subprocess.run(["python", "Positions.py"])