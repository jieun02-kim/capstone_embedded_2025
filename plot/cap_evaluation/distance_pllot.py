import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("evaluation_results.csv")

plt.figure(figsize=(10,5))
plt.plot(df['distance_cmd'], label='cmd distance')
plt.plot(df['distance_odom'], label='odom distance')
plt.plot(df['need_dist'], label='need distance')
plt.legend()
plt.xlabel('steps')
plt.ylabel('distance (m)')
plt.title('Distance Comparison')
plt.show()
