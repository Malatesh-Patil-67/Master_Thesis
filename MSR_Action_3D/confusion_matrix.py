#confusion matrix

action_names = [
    "high arm wave",
    "horizontal arm wave",
    "hammer",
    "hand catch",
    "forward punch",
    "high throw",
    "draw x",
    "draw tick",
    "draw circle",
    "hand clap",
    "two hand wave",
    "side-boxing",
    "bend",
    "forward kick",
    "side kick",
    "jogging",
    "tennis swing",
    "tennis serve",
    "golf swing",
    "pick up & throw"
]


predictions = np.array(test_predictions)
true_labels = np.array(test_targets)

# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, predictions)

# Define custom colormap from black to orange
colors = ["black", "orange"]
cmap = LinearSegmentedColormap.from_list("Custom", colors)

# Compute confusion matrix as percentages
conf_matrix_percentage = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

# Plot confusion matrix with actual action names
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(conf_matrix_percentage, fmt='.1f', cmap=cmap, xticklabels=action_names, yticklabels=action_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()

# Save the figure
plt.savefig('confusion_matrix_percentage.png', dpi=300)

# Show the plot
plt.show()

plt.close()