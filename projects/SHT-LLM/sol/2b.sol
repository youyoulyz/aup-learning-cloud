#6.1

layer1 = nn.Linear(20, 30).to(device)
layer2 = nn.Linear(30, 40).to(device)

print("--- Network Construction ---")
print(f"Layer 1: {layer1}")
print(f"Layer 2: {layer2}")

batch_size = 128
x = torch.randn(batch_size, 20, device=device)
y1 = layer1(x)
y2 = layer2(y1)

print("\n--- Forward Pass ---")
print(f"Input shape:      {x.shape}")
print(f"Shape after L1:   {y1.shape}")
print(f"Final shape (L2): {y2.shape}")

params1 = sum(p.numel() for p in layer1.parameters())
params2 = sum(p.numel() for p in layer2.parameters())
print("\n--- Parameter Analysis ---")
print(f"Layer 1 Params: {params1:,}")
print(f"Layer 2 Params: {params2:,}")
print(f"Total Params:   {params1 + params2:,}")

loss = y2.mean()
loss.backward()

print("\n--- Gradient Flow ---")
print(f"Grads exist for L1 weights: {layer1.weight.grad is not None}")
print(f"Grads exist for L2 weights: {layer2.weight.grad is not None}")
assert layer2.weight.grad is not None, "Gradients for layer2 are missing!"
print("Gradient check passed!")

#6.2

x_large = torch.randn(128, 4096, 30, 20, device=device)

print("--- Large Tensor Processing ---")
print(f"Large tensor shape: {x_large.shape}")
print(f"Memory usage: {x_large.numel() * x_large.element_size() / (1024**2):.1f} MB")

y1_large = layer1(x_large)
y2_large = layer2(y1_large)

print("\n--- Direct Application on 4D Tensor ---")
print(f"Shape after L1: {y1_large.shape}")
print(f"Final shape (L2): {y2_large.shape}")

expected_shape = (128, 4096, 30, 40)
assert y2_large.shape == expected_shape, "The final shape is incorrect!"
print(f"\nShape verification successful!")

#6.3
x_reshape = torch.randn(128, 4096, 30, 20, device=device)
b, s, h, w = x_reshape.shape
print(f"--- Tensor Reshaping Strategies ---")
print(f"Original shape: {x_reshape.shape}")

x_flat = x_reshape.view(b * s * h, w)
print(f"\nFlattened shape for processing: {x_flat.shape}")

y1_flat = layer1(x_flat)
y2_flat = layer2(y1_flat)
print(f"Shape after processing: {y2_flat.shape}")

y2_reshaped = y2_flat.view(b, s, h, 40)
print(f"\nReshaped back to 4D: {y2_reshaped.shape}")

y2_large_verify = layer2(layer1(x_reshape))
results_are_identical = torch.allclose(y2_large_verify, y2_reshaped)
assert results_are_identical, "Reshaping method did not match direct method!"
print("\nVerification successful: Reshaping yields identical results.")

x_auto = x_reshape.view(-1, w)
print(f"\nReshaping with -1: {x_auto.shape}")
assert x_auto.shape == x_flat.shape
print("Automatic dimension calculation successful.")

#8.2
class LeNet(nn.Module):
    """LeNet-5 variant for FashionMNIST classification"""
    
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)    # 28×28×1 → 24×24×6
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)   # 12×12×6 → 8×8×16
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # Flattened conv output → 120
        self.fc2 = nn.Linear(120, 84)          # 120 → 84
        self.fc3 = nn.Linear(84, num_classes)  # 84 → 10 classes
        
        # Pooling layer (reused)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        """Forward pass through the network"""
        # Feature extraction
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 → ReLU → Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 → ReLU → Pool
        
        # Flatten for fully connected layers
        x = x.view(-1, 16 * 4 * 4)  # Flatten: [batch, 16, 4, 4] → [batch, 256]
        
        # Classification layers
        x = F.relu(self.fc1(x))  # FC1 → ReLU
        x = F.relu(self.fc2(x))  # FC2 → ReLU
        x = self.fc3(x)          # Final layer (no activation - raw logits)
        
        return x

#8.3
def train_epoch(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        predictions = model(X)
        loss = loss_fn(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct_predictions += (predictions.argmax(1) == y).sum().item()
        if batch_idx % 100 == 0:
            current_samples = batch_idx * len(X)
            print(f"  Batch {batch_idx:3d}: Loss = {loss.item():.6f} | Progress: [{current_samples:5d}/{size:5d}] ({100.*current_samples/size:.1f}%)")
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / size
    return avg_loss, accuracy

def evaluate_model(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            predictions = model(X)
            total_loss += loss_fn(predictions, y).item()
            correct_predictions += (predictions.argmax(1) == y).sum().item()
    avg_loss = total_loss / num_batches
    accuracy = correct_predictions / size
    return avg_loss, accuracy
