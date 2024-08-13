import matplotlib.pyplot as plt
import numpy as np

# Metric categories
categories = ['Precision', 'Recall', 'F1-Score']

# Values for each category for Class 0 (Non-Happiness) and Class 1 (Happiness)
class_0 = [0.73, 0.90, 0.81]  # Non-Happiness
class_1 = [0.70, 0.42, 0.52]  # Happiness

# Position of bars on x-axis
ind = np.arange(len(categories))

# Width of a bar 
width = 0.35       

plt.figure(figsize=(10, 5))

# Plotting
bar1 = plt.bar(ind, class_0, width, label='Non-Happiness', color='blue')  # Blue for Non-Happiness
bar2 = plt.bar(ind + width, class_1, width, label='Happiness', color='green')  # Green for Happiness

# Adding Xticks
plt.xlabel('Metrics', fontsize=12)
plt.ylabel('Scores', fontsize=12)
plt.title('Model Performance by Class')
plt.xticks(ind + width / 2, categories)
plt.legend(loc='best')
plt.show()
