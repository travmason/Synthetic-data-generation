import json
import matplotlib.pyplot as plt

# Load JSON data from file
with open('gpt3_logs/score.log') as file:
    json_data = file.read()

# Split the file content by consecutive closing and opening braces
json_objects = json_data.split('}{')
# json_objects = ['{' + obj + '}' for obj in json_objects]  # Add braces to each object

# Parse each JSON object and extract data for visualization
x_labels = []
y_values = []
for obj in json_objects:
    data = json.loads(obj)
    for key, value in data.items():
        x_labels.append(key)
        y_values.append(value)

# Create a bar plot
plt.bar(x_labels, y_values)
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.xlabel("Skills")
plt.ylabel("Scores")
plt.title("Skills Evaluation")
plt.tight_layout()  # Adjust the layout to prevent label cutoffs
plt.show()