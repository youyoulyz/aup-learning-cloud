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


# 4.3