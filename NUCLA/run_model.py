import model
import nucla_dataset
import eval
import confusion_matrix
import accuracy_vs_epochs
import t_sne_plot

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the root directory of your dataset
train_root_dir = 'pathtotraindatasetfolder'

test_root_dir = 'pathtotestdatasetfolder'

# Create an instance of the training dataset
train_dataset = MyCustomDataset(train_root_dir, mode='train')

# Create an instance of the testing dataset
test_dataset = MyCustomDataset(test_root_dir, mode='test')

# Set the batch size
batch_size = 32

# Create data loaders

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

train_dataset_length = len(train_dataset)

test_dataset_length = len(test_dataset)



# Set the number of output classes
num_classes = 10 # Number of classes in dataset
emb_dim = num_classes

# Set the hyperparameters
num_frame =91
num_joints = 20
in_chans = 3
embed_dim_ratio = 64
num_frame_kept=27
num_coeff_kept=27
depth = 4
num_heads = 8
mlp_ratio = 2.
qkv_bias = True
qk_scale = None
drop_rate = 0.
attn_drop_rate = 0.
drop_path_rate = 0.2



# Create an instance of the PoseTransformer model
model = PoseTransformerV2(num_frame=num_frame, num_joints=num_joints, in_chans=in_chans, embed_dim_ratio=embed_dim_ratio,
                          num_frame_kept=num_frame_kept,num_coeff_kept=num_coeff_kept,
                        depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                        num_classes =num_classes)

# Move the model to the device
model = model.to(device)

# Set the optimizer and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Set the loss function
criterion = nn.CrossEntropyLoss()

# Set the number of training epochs
num_epochs = 100


# Create a label encoder
label_encoder = LabelEncoder()

# Fit label encoder on the training labels
label_encoder.fit(train_dataset.labels)

# Convert training labels to numerical values
train_labels_encoded = label_encoder.transform(train_dataset.labels)

# Convert the encoded labels to tensors
train_labels_tensor = torch.tensor(train_labels_encoded, dtype=torch.long).to(device)


train_accuracies = []
train_losses = []

test_accuracies = []
test_losses = []

train_predictions = []
train_targets = []
train_lab_pred =[]
train_lab=[]

test_predictions = []
test_targets = []
test_lab_pred =[]
true_lab=[]


# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    train_loss = 0.0
    train_correct = 0

    all_embeddings_tsne = []

    train_predictions = []
    train_targets = []
    train_lab_pred =[]
    train_lab=[]


    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)

        labels_encoded = label_encoder.transform(labels)  # Convert current batch labels to numerical values
        labels_tensor = torch.tensor(labels_encoded, dtype=torch.long).to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        embeddings_train = outputs

        embeddings_tsne_train = embeddings_train.view(-1,1,emb_dim)

        all_embeddings_tsne.append(embeddings_tsne_train)

        # Compute the loss
        loss = criterion(outputs, labels_tensor)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Compute the training accuracy
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels_tensor).sum().item()

        # Update the training loss
        train_loss += loss.item() * inputs.size(0)



        train_predictions.extend(predicted.cpu().numpy())
        train_targets.extend(labels_encoded)

        train_lab_pred.append(labels_tensor)

        train_lab.append(labels)

    # Compute the average training loss and accuracy
    train_loss = train_loss / len(train_dataset)
    train_accuracy = train_correct / len(train_dataset)
    train_accuracies.append(train_accuracy)
    train_losses.append(train_loss)

    # Print the training loss and accuracy for each epoch
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')





    # Testing loop
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    test_correct = 0


    all_test_embeddings = []

    test_predictions = []
    test_targets = []
    test_lab_pred =[]
    true_lab=[]

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)

            labels_encoded = label_encoder.transform(labels)  # Convert current batch labels to numerical values
            labels_tensor = torch.tensor(labels_encoded, dtype=torch.long).to(device)


            # Forward pass
            outputs = model(inputs)

            embeddings_test = outputs

            embeddings_tsne_test = embeddings_test.view(-1,1,emb_dim)

            all_test_embeddings.append(embeddings_tsne_test)

            # Compute the loss
            loss = criterion(outputs, labels_tensor)

            # Compute the testing accuracy
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels_tensor).sum().item()

            # Update the testing loss
            test_loss += loss.item() * inputs.size(0)

            # Collect predictions and targets for computing metrics


            test_predictions.extend(predicted.cpu().numpy())
            test_targets.extend(labels_encoded)

            test_lab_pred.append(labels_tensor)

            true_lab.append(labels)

    # Compute the average testing loss and accuracy
    test_loss = test_loss / len(test_dataset)
    test_accuracy = test_correct / len(test_dataset)
    test_accuracies.append(test_accuracy)
    test_losses.append(test_loss)

    # Print the testing loss and accuracy for each epoch
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
