import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Process and view the scores in a folder. Defaults to the last run.")
parser.add_argument('-i', '--input', type=str, help="Optional. Input folder. Defaults to the last run.")
parser.add_argument('-c', '--colour', type=str, help="Optional. Colour palette from Seaborn available options. Defaults to muted. Available paletes: deep, muted, bright, pastel, dark, colorblind")

args = parser.parse_args()

directory = 'gpt3_logs'

# Create a list of all the files in the current directory
# we're going to use this to create a new directory for the current run
# filelist = filter(lambda x: (x.endswith('.run')), os.listdir(directory))

filelist = filter(lambda x: (x.endswith('.run')), os.listdir(directory))

#check arguments to see if we have a specified directory to use
if args.input:
    if not os.path.exists(os.path.join(directory, f"{args.input}.run")):
        print(f'Error: Directory {args.input}.run does not exist.')
        exit(1)
    highest_number = args.input
else:
    # Find the highest numbered directory
    highest_number = 0
    for file in filelist:
        try:
            number = int(file.rstrip('.run'))
            if number > highest_number:
                highest_number = number
        except ValueError:
            pass  # Ignore if the file name is not a number

#set the new working directory based on the working directory name
directory = os.path.join(directory, f"{highest_number}.run")

# Directory path containing the JSON files
directory_path = directory

# Directory path containing the JSON files
directory_path = directory
log_file = directory_path.split('/')[1].split('.')[0]
log_file = log_file + '_score.log'

data = []

# Open the file and read the data
with open(os.path.join(directory_path, log_file), 'r') as file:
    data = json.load(file)

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

# Calculate the standard error for each category
standard_errors = [np.std(data_by_category[label]) / np.sqrt(len(data_by_category[label])) for label in labels]

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
legend_handles = [mpatches.Patch(color=colors[i], 
                                 label=f"{i+1}. {labels[i]}\nMean={averages[i]:.2f}\nSE={standard_errors[i]:.2f}") 
                  for i in range(len(labels))]

# Display the legend
ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_ylabel('Scores')
ax.set_title('Scores by Category')
#set the y axis values
ax.set_ylim([0, 100])

# Adjust layout
plt.tight_layout()
plt.savefig(f'{highest_number}boxplot.png', dpi=300, bbox_inches='tight')
# Display the plot
#plt.show()

###
# Bar chart
###
# Create a new figure for the bar chart
fig, ax = plt.subplots(figsize=(10, 6))

# Get a color palette from seaborn
if args.colour:
    colors = sns.color_palette(args.colour, n_colors=len(labels))
else:
    colors = sns.color_palette("muted", n_colors=len(labels))

# Create the bar chart
bars = ax.barh(labels, averages, xerr=standard_errors, align='center', alpha=0.7, color=colors, capsize=5)

# Annotate each bar with its standard error value
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.annotate(f'SE: {standard_errors[i]:.2f}',
                xy=(width - 2*standard_errors[i], bar.get_y() + bar.get_height() / 2),
                xytext=(40, 0),  # 15 points horizontal offset to the left
                textcoords="offset points",
                ha='center', va='baseline')

# Labeling and presentation
ax.set_xlabel('Scores')
ax.set_title('Scores by Category with Standard Errors')
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)

ax.spines[['right', 'top', 'bottom']].set_visible(False) 
ax.xaxis.set_visible(False)

ax.bar_label(bars, padding=-85, color='black', 
             fontsize=12, label_type='center', fmt='%.1f',
            fontweight='bold')

# Display the plot
plt.tight_layout()
plt.savefig(f'{highest_number}barplot.png', dpi=300, bbox_inches='tight')
plt.show()