def moving_average(data, window_size):
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, 'valid')

def plot_accuracy_vs_epochs(train_accuracies, test_accuracies, num_epochs, window_size=5):
    smoothed_train_accuracies = moving_average(train_accuracies, window_size)
    smoothed_test_accuracies = moving_average(test_accuracies, window_size)
    epochs = np.arange(1, num_epochs + 1)[:len(smoothed_train_accuracies)]

    plt.figure(figsize=(8, 6))  # Adjust figure size as needed
    plt.plot(epochs, smoothed_train_accuracies, color='orange', linestyle='-', label='Training Accuracy')  # Adjust line color and label
    plt.plot(epochs, smoothed_test_accuracies, color='blue', linestyle='-', label='Testing Accuracy')  # Adjust line color and label
    plt.xlabel('Epochs', fontsize=12)  # Adjust font size as needed
    plt.ylabel('Accuracy', fontsize=12)  # Adjust font size as needed
    plt.title('Training and Testing Accuracy vs Epochs', fontsize=14, fontweight='bold')  # Adjust font size and weight as needed
    plt.xticks(fontsize=10)  # Adjust font size of x-axis ticks as needed
    plt.yticks(fontsize=10)  # Adjust font size of y-axis ticks as needed
    plt.legend()  # Add legend
    plt.ylim(0.0, 1.0)  # Set y-axis limit to start from 0.4
    #plt.grid(True, linestyle='--', alpha=0.7)  # Add grid lines with custom style and transparency
    plt.gca().set_facecolor('black')  # Set background color to black
    plt.tight_layout()  # Adjust layout to prevent overlap of labels
    plt.savefig('train_test_accuracy_vs_epochs.png', dpi=300)  # Save plot with high resolution
    plt.close()  # Close the figure to release memory resources


plot_accuracy_vs_epochs(train_accuracies, test_accuracies, num_epochs)