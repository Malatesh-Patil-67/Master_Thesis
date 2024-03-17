average_accuracy = round(np.mean(accuracies), 4)
average_precision = round(np.mean(precision), 4)
average_recall = round(np.mean(recall), 4)
average_f1 = round(np.mean(f1), 4)


# Define the data for the table
data = [
    ["Average Accuracy", average_accuracy],
    ["Average Precision", average_precision],
    ["Average Recall", average_recall],
    ["Average F1 Score", average_f1]

]

# Define column headers
headers = ["Metrics", "Value"]

# Print the table
print(tabulate(data, headers=headers, tablefmt="grid"))