import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

data = []

# Open the file and read the data
with open('gpt3_logs/score.log', 'r') as file:
    data = json.load(file)

# Convert the JSON strings to Python dictionaries
# data = [json.loads(js) for js in json_strings]

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

###
# Boxplot
###
# Create a figure and a set of subplots
fig, ax = plt.subplots(figsize=(10, 6))

# Convert the data_by_category values to a list of lists for boxplot
scores = list(data_by_category.values())

# Create a boxplot and fill boxes with color
bp = ax.boxplot(scores, patch_artist=True)

# Set the x-tick labels to numbers and rotate them
ax.set_xticklabels(list(range(1, len(labels) + 1)))

# Define a list of colors for the boxes
colors = ['red', 'green', 'blue', 'cyan', 'magenta']

# Color the boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Create legend handles and labels
legend_handles = [mpatches.Patch(color=colors[i], label=f"{i+1}. {labels[i]}") for i in range(len(labels))]

# Display the legend
ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')

ax.set_ylabel('Scores')
ax.set_title('Scores by Category')

#set the y axis values
ax.set_ylim([0, 100])

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()


###
# Average across categories in the rubric.
###
# Define a list of colors for the bars
colors = ['red', 'green', 'blue', 'cyan', 'magenta']

# Create a figure and a set of subplots
fig, ax = plt.subplots(figsize=(10, 6))

# Plot averages with error bars representing variance, assign a color to each bar
bars = ax.bar(range(len(labels)), averages, yerr=variances, align='center', alpha=0.5, 
              ecolor='black', capsize=10, color=colors)
ax.set_ylabel('Average Scores')
ax.set_title('Average Scores by Category with Variance')
ax.yaxis.grid(True)

# Replace x-ticks with numbers
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(range(1, len(labels) + 1))

# Create legend handles and labels
legend_handles = [mpatches.Patch(color=colors[i], label=f"{i+1}. {labels[i]}") for i in range(len(labels))]

# Display the legend
ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout
plt.tight_layout()

# Display the plot
#plt.show()

###
# All the data as subplots on a bar graph - for visual check of variance, etc.
###
# The x position of the bars
x = np.arange(len(labels))
print(f'x: {x}')

# Create subplots to have multiple bars
fig, ax = plt.subplots()

print(f'len(data): {len(data)}')
# Plot data
for i in range(len(data)):
    print(f'i: {i}')
    ax.bar(x + i/len(data), scores[i], width=1/len(data), label=f'Data {i+1}')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by category and data')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Rotate the x-axis labels for better visibility
plt.xticks(rotation=45)
plt.tight_layout() # # Adjust layout to avoid overlapping labels

# Show the plot
plt.show()
