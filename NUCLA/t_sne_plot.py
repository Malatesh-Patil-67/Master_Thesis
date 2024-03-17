scaler = StandardScaler()

all_embeddings_tsne_test = torch.cat(all_test_embeddings, dim=0)

embeds = all_embeddings_tsne_test

targets = torch.cat(test_lab_pred, dim=0)

custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


# Convert anchor_embeddings to numpy array
embeddings_np = embeds.cpu().numpy()  # Assuming anchor_embeddings is a PyTorch tensor

#Flatten the anchor embeddings to a 2D array
flattened_embeddings = embeds.view(embeds.size(0), -1).cpu().numpy()

flattened_embeddings_scaled = scaler.fit_transform(flattened_embeddings)


# Get the labels for the tsne embeddings
tsne_labels = targets.cpu().numpy()


perplexity_values = [5,40,50]
iteration_values = [800,1000,1200]


plt.figure(figsize=(9, 9))
plot_num = 1
num= 1

cmap = ListedColormap(custom_colors[:len(np.unique(tsne_labels))])

for perplexity in perplexity_values:
    for iterations in iteration_values:
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=iterations, random_state=42)
        tsne_embeddings = tsne.fit_transform(flattened_embeddings_scaled)
        plt.subplot(len(perplexity_values), len(iteration_values), plot_num)
        plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=tsne_labels, cmap=cmap , s=0.1)
        plt.title(f'Perplexity: {perplexity}, Iterations: {iterations}')
        plt.show()
        plt.xticks([])
        plt.yticks([])
        plt.savefig('test_100_tsne_plot' + str(num) + '.png', dpi=300)
        plot_num += 1
        num = num+1

plt.close()




action_names = [
    "Pick up with one hand", "Pick up with two hands", "Drop trash", "Walk around", "Sit down",
    "Stand up", "Donning", "Doffing", "Throw", "Carry"
]

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Plot each color with its corresponding action name using circles
for i, (color, action) in enumerate(zip(custom_colors, action_names)):
    ax.scatter([i]*2, [0, 1], color=color, label=action, marker='o', s=200)  # Use circles as markers

# Hide the axes
ax.axis('off')

# Add legend with adjusted font size
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='large')  # You can adjust font size here

# Save the plot as an image
plt.savefig('color_action_plot.png', bbox_inches='tight')

# Show plot
plt.tight_layout()
plt.show()

plt.close()