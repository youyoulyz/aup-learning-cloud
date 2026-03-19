#1.3

# Assume the following tensors are pre-defined from a previous cell:
# x: input tensor of shape (10, 5)
# w: weight tensor of shape (5, 3)
# y_manual_matmul: the result from the manual x @ w calculation
# device: the target device ('cuda' or 'cpu')

print("=== PyTorch Linear Layer Solution ===")

# --- 1. Create the linear layer ---
# Define a linear layer with 5 input features and 3 output features.
# Make sure to disable the bias and move the layer to the correct device.
linear = torch.nn.Linear(in_features=5, out_features=3, bias=False).to(device)

print(f"Linear layer created. Weight shape: {linear.weight.shape}")
print(f"Layer is on device: {linear.weight.device}")
print(f"Layer bias is: {linear.bias}") # Should be None

# --- 2. Set the weights manually ---
# The layer's `weight` attribute has a shape of (out_features, in_features).
# Our `w` tensor has a shape of (in_features, out_features).
# You will need to transpose `w` before assigning it.
linear.weight.data = w.T # Note: Linear layers store weights transposed

print(f"\nOriginal weight matrix w (shape {w.shape}):\n{w}")
print(f"Linear layer weight after setting (shape {linear.weight.shape}):\n{linear.weight}")

# --- 3. Apply the linear transformation ---
# Pass the input tensor `x` through the layer to compute the output.
y_linear = linear(x)

print(f"\nLinear layer output shape: {y_linear.shape}")
print(f"First 3 outputs:\n{y_linear[:3]}")

# --- 4. Verify equivalence ---
# Use torch.allclose() to confirm the results are the same.
# This should return True if your implementation is correct.
are_equivalent = torch.allclose(y_manual_matmul, y_linear)
print(f"\nManual matmul vs Linear layer outputs are equivalent: {are_equivalent}")

# Assert that the check passes
assert are_equivalent, "Outputs do not match! Please review your code."

print("\nGreat job! You've successfully used a PyTorch Linear layer.")

#2.1

batch_size = 10
in_features = 20
hidden_features = 8
out_features = 3

input_data = torch.randn(batch_size, in_features, device=device)
target = torch.randn(batch_size, out_features, device=device)
reg_strength = 0.01

w1 = torch.randn(in_features, hidden_features, requires_grad=True, device=device)
b1 = torch.randn(hidden_features, requires_grad=True, device=device)

w2 = torch.randn(hidden_features, out_features, requires_grad=True, device=device)
b2 = torch.randn(out_features, requires_grad=True, device=device)

hidden_input = input_data @ w1 + b1
hidden_output = F.relu(hidden_input)
prediction = hidden_output @ w2 + b2

mse_loss = torch.mean((prediction - target)**2)
l2_reg = reg_strength * (torch.sum(w1**2) + torch.sum(w2**2))
total_loss = mse_loss + l2_reg

print(f"Forward Pass Completed:")
print(f"  Prediction shape: {prediction.shape}")
print(f"  Total Loss: {total_loss.item():.4f}")

total_loss.backward()

print(f"\nBackward Pass Completed. Gradients are now available.")
print(f"  Gradient for w1 has shape: {w1.grad.shape}")
print(f"  Gradient for b1 has shape: {b1.grad.shape}")
print(f"  Gradient for w2 has shape: {w2.grad.shape}")
print(f"  Gradient for b2 has shape: {b2.grad.shape}")
print(f"  Norm of w1 gradient: {w1.grad.norm().item():.4f}")

#4.2

class ModernLeNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self._conv_output_size = self._get_conv_output_size(in_channels)

        self.classifier = nn.Sequential(
            nn.Linear(self._conv_output_size, 120),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

    def _get_conv_output_size(self, in_channels):
        dummy_input = torch.randn(1, in_channels, 28, 28)
        output = self.features(dummy_input)
        flattened_size = int(torch.flatten(output, 1).size(1))
        print(f"Dynamically calculated flattened size for FC layer: {flattened_size}")
        return flattened_size

    def forward(self, x):
        features_out = self.features(x)
        flattened = torch.flatten(features_out, 1)
        final_output = self.classifier(flattened)
        return final_output

#4.3

# 4.3 Complete Training Pipeline

print("=== Training Setup ===")
model = ModernLeNet(num_classes=10).to(device)
#model = ModernLeNet(num_classes=10).to(device)

# --- 1. Define Loss Function and Optimizer ---
# We'll use Cross-Entropy Loss for this multi-class classification problem.
criterion = nn.CrossEntropyLoss()
# For the optimizer, try using Adam, a popular and effective alternative to SGD.
# Look up the documentation for torch.optim.Adam.
# torch.optim.Adam takes the model parameters and a learning rate as arguments.
# You can call by passing (model.parameters(), lr ) to the optimizer to tell it which parameters to update.
#  A learning rate of 1e-3 is a good start.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f"Loss function: {criterion}")
print(f"Optimizer: {optimizer}")

def train_epoch(dataloader, model, loss_fn, optimizer, device):
    """Trains the model for one epoch."""
    model.train() # Set the model to training mode
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # --- 2. Forward Pass ---
        # Get the model's predictions (logits) for the input batch.
        # Then, calculate the loss between the predictions and the true labels.
        predictions = model(X)
        # The loss function will compare the predicted logits with the true labels `y`.
        # For CrossEntropyLoss, `predictions` should be raw logits (not probabilities), and `y` should be the class indices.
        # Math for loss: loss = loss_fn(predictions, y)
        loss = loss_fn(predictions, y)

        # --- 3. Backward Pass & Optimizer Step ---
        # First, clear the gradients from the previous step.
        # Then, perform backpropagation to compute gradients.
        # Finally, update the model's weights using the optimizer.
        optimizer.zero_grad() # Zero gradients
        loss.backward()  # Backpropagate
        optimizer.step()  # Update weights

        # --- 4. Track Statistics ---
        # We track loss and accuracy to monitor training progress.
        total_loss += loss.item()
        # Find the index of the max logit for each sample to get the predicted class.
        # Math for predicted class: _, predicted_classes = torch.max(predictions, dim=1)
        _, predicted_classes = torch.max(predictions, 1)
        total_samples += y.size(0)
        correct_predictions += (predicted_classes == y).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct_predictions / total_samples
    return avg_loss, accuracy

def test_epoch(dataloader, model, loss_fn, device):
    """Evaluates the model on the test dataset."""
    model.eval() # Set the model to evaluation mode
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    # In evaluation, we don't need to compute gradients.
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # --- 5. Evaluation Forward Pass ---
            # This is similar to training, but without the backward pass.
            # Get predictions and calculate the loss.
            predictions =  model(X)
            # Calculate the loss for this batch using the same loss function as training.
            loss = loss_fn(predictions, y)

            # --- 6. Track Evaluation Statistics ---
            total_loss += loss.item()
            _, predicted_classes = torch.max(predictions, 1)
            total_samples += y.size(0)
            correct_predictions += (predicted_classes == y).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct_predictions / total_samples
    return avg_loss, accuracy

# --- The Main Training Loop (Provided for you) ---
print(f"\n=== Starting Training on {device} ===")
epochs = 5
train_losses, train_accs = [], []
test_losses, test_accs = [], []

for epoch in range(epochs):
    print(f"\n--- Epoch {epoch+1}/{epochs} ---")

    train_loss, train_acc = train_epoch(train_dataloader, model, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    test_loss, test_acc = test_epoch(test_dataloader, model, criterion, device)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%")

print(f"\n=== Training Complete! ===")
print(f"Final Test Accuracy: {test_accs[-1]:.2f}%")

print(f"\nKey Learning Points:")
print(f"   • Forward pass: Data flows through network layers")
print(f"   • Loss computation: Measures prediction errors")
print(f"   • Backward pass: Computes gradients via chain rule")
print(f"   • Parameter update: Moves in direction of negative gradient")
print(f"   • This is the foundation of ALL deep learning!")
print(f"   • Same principles apply to LLMs, just different architectures")
