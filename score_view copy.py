import json
import matplotlib.pyplot as plt
import numpy as np

data = []

# Open the file and read the data
with open('gpt3_logs/score.log', 'r') as file:
    json_strings = file.read()

# Convert the JSON strings to Python dictionaries
data = [json.loads(js) for js in json_strings.split("\n") if js]

# Get the keys from the first dictionary to use as labels
labels = list(data[0].keys())

# Create a 2D list where each inner list represents the scores for one dictionary
scores = [[d[label] for label in labels] for d in data]

# First, let's reorganize the data by category
data_by_category = {label: [d[label] for d in data] for label in labels}

# Now let's calculate the averages, variance, and high/low values for each category
averages = [np.mean(data_by_category[label]) for label in labels]
variances = [np.var(data_by_category[label]) for label in labels]
max_values = [np.max(data_by_category[label]) for label in labels]
min_values = [np.min(data_by_category[label]) for label in labels]

# Create subplots to have multiple bars
fig, ax = plt.subplots()

# The x position of the bars
x = np.arange(len(labels))

# Plot data
for i in range(len(data)):
    ax.bar(x + i/len(data), scores[i], width=1/len(data), label=f'Data {i+1}')

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by category and data')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Rotate the x-axis labels for better visibility
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust layout to avoid overlapping labels

# Show the plot
plt.show()
