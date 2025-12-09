import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("evaluation_log.csv")

plt.figure(figsize=(10,5))
plt.plot(df['cmd_linear'], label='cmd_linear')
plt.plot(df['cmd_angular'], label='cmd_angular')
plt.plot(df['odom_x'], label='odom_x')
plt.legend()
plt.xlabel('steps')
plt.ylabel('distance (m)')
plt.title('Distance Comparison')
plt.show()
