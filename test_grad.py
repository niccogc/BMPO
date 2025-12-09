import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F
import quimb.tensor as qt
from tqdm import tqdm
import matplotlib.pyplot as plt

# Use float32 for CPU stability
torch.set_default_dtype(torch.float32)

# --- 1. User's Data Preprocessing (CPU Version) ---

print("Loading and processing data on CPU...")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load Raw Data
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader_raw = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Process Train
train_samples = []
train_labels = []
for images, labels in train_loader_raw:
    train_samples.append(images)
    train_labels.append(labels)
xinp_train = torch.cat(train_samples, dim=0)
y_train = torch.cat(train_labels, dim=0)

KERNEL_SIZE = 4
STRIDE = 4

# Unfold and Transpose -> (Batch, Num_Patches, Pixels_Per_Patch)
xinp_train = F.unfold(xinp_train, kernel_size=(KERNEL_SIZE,KERNEL_SIZE), stride=(STRIDE,STRIDE), padding=0).transpose(-2, -1)

# Pad Patches (Dim -2): 49 -> 50
xinp_train = torch.cat((xinp_train, torch.zeros((xinp_train.shape[0], 1, xinp_train.shape[2]))), dim=-2)
# Pad Pixels (Dim -1): 16 -> 17
xinp_train = torch.cat((xinp_train, torch.zeros((xinp_train.shape[0], xinp_train.shape[1], 1))), dim=-1)

# Set corner
xinp_train[..., -1, -1] = 1.0
# Labels (CPU)
y_train_indices = y_train

# Process Test
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
test_loader_raw = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
test_samples = []
test_labels = []
for images, labels in test_loader_raw:
    test_samples.append(images)
    test_labels.append(labels)
xinp_test = torch.cat(test_samples, dim=0)
y_test = torch.cat(test_labels, dim=0)

xinp_test = F.unfold(xinp_test, kernel_size=(KERNEL_SIZE,KERNEL_SIZE), stride=(STRIDE,STRIDE), padding=0).transpose(-2, -1)
xinp_test = torch.cat((xinp_test, torch.zeros((xinp_test.shape[0], 1, xinp_test.shape[2]))), dim=-2)
xinp_test = torch.cat((xinp_test, torch.zeros((xinp_test.shape[0], xinp_test.shape[1], 1))), dim=-1)
xinp_test[..., -1, -1] = 1.0
y_test_indices = y_test

print(f"Data shape: {xinp_train.shape}") # Expected: (60000, 50, 17)

# Create DataLoaders
BATCH_SIZE = 100
train_ds = torch.utils.data.TensorDataset(xinp_train, y_train_indices)
test_ds = torch.utils.data.TensorDataset(xinp_test, y_test_indices)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


# --- 2. The TN Model Class ---

class TNModel(nn.Module):
    def __init__(self, tn_pixels, tn_patches, output_dims, input_dims):
        super().__init__()
        
        params_pixel, self.skeleton_pixels = qt.pack(tn_pixels)
        params_patches, self.skeleton_patches = qt.pack(tn_patches)

        self.torch_params_pixels = nn.ParameterDict({
            str(i): nn.Parameter(initial) for i, initial in params_pixel.items()
        })
        self.torch_params_patches = nn.ParameterDict({
            str(i): nn.Parameter(initial) for i, initial in params_patches.items()
        })
        
        self.input_dims = input_dims
        self.output_dims = output_dims

    def forward(self, x):
        # Unpack parameters
        params_pixels = {int(i): p for i, p in self.torch_params_pixels.items()}
        pixels = qt.unpack(params_pixels, self.skeleton_pixels)
        
        params_patches = {int(i): p for i, p in self.torch_params_patches.items()}
        patches = qt.unpack(params_patches, self.skeleton_patches)
        
        # Construct Input Nodes
        input_nodes_list = self.construct_nodes(x)
        input_tn = qt.TensorNetwork(input_nodes_list)

        # Contract: Pixels & Patches & Input
        tn = pixels & patches & input_tn
        
        # Contract and return raw data
        out = tn.contract(output_inds=self.output_dims, optimize='auto-hq')
        return out.data

    def construct_nodes(self, x):
        input_nodes = []
        for i in self.input_dims:
            # Data shape is (Batch, Patches, Pixels) -> (B, 50, 17)
            # Indices: ["s", "patches_dim", "pixels_dim"]
            a = qt.Tensor(x, inds=["s", f"{i}_patches", f"{i}_pixels"], tags=f"Input_{i}")
            input_nodes.append(a)
        return input_nodes


# --- 3. Manual Tensor Construction ---

# Derived Dimensions from your data
DIM_PATCHES = xinp_train.shape[1] # 50
DIM_PIXELS = xinp_train.shape[2]  # 17

bond_dim = 8

# Helper for init
def init_data(*shape):
    return torch.randn(*shape) * 0.1

# === A. Pixels MPS ===
# Interactions with Inner Dimension (Size 17)
pixels_mps_list = [
    qt.Tensor(data=init_data(1, DIM_PIXELS, bond_dim), inds=["_null_p_l", "0_pixels", "bond_p_01"], tags=["PIXELS", "0"]),
    qt.Tensor(data=init_data(bond_dim, DIM_PIXELS, bond_dim, 10), inds=["bond_p_01", "1_pixels", "bond_p_12", "class_out"], tags=["PIXELS", "1", "OUTPUT"]),
    qt.Tensor(data=init_data(bond_dim, DIM_PIXELS, 1), inds=["bond_p_12", "2_pixels", "_null_p_r"], tags=["PIXELS", "2"])
]

# === B. Patches MPS ===
# Interactions with Outer Dimension (Size 50)
patches_mps_list = [
    qt.Tensor(data=init_data(bond_dim, DIM_PATCHES, bond_dim), inds=["r1", "0_patches", "bond_pt_01"], tags=["PATCHES", "0"]),
    qt.Tensor(data=init_data(bond_dim, DIM_PATCHES, bond_dim), inds=["bond_pt_01", "1_patches", "bond_pt_12"], tags=["PATCHES", "1"]),
    qt.Tensor(data=init_data(bond_dim, DIM_PATCHES, bond_dim), inds=["bond_pt_12", "2_patches", "r1"], tags=["PATCHES", "2"])
]

# --- 4. Initialize & Train ---

device = torch.device("cpu")
input_labels = ["0", "1", "2"] 

model = TNModel(
    tn_pixels=qt.TensorNetwork(pixels_mps_list), 
    tn_patches=qt.TensorNetwork(patches_mps_list), 
    output_dims=["s", "class_out"],
    input_dims=input_labels
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# --- Training Loop ---

history = {'epoch': [], 'loss': [], 'accuracy': []}
epochs = 5

print(f"Model initialized on {device}. Input: {DIM_PATCHES} Patches x {DIM_PIXELS} Pixels.")

def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_loss = running_loss / len(train_loader)
    acc = evaluate(test_loader)
    print(f"Epoch {epoch+1} Test Accuracy: {acc*100:.2f}%")
    
    history['epoch'].append(epoch + 1)
    history['loss'].append(avg_loss)
    history['accuracy'].append(acc * 100)

# --- Visualization ---

plt.figure(figsize=(10, 5))
ax1 = plt.gca()
l1, = ax1.plot(history['epoch'], history['loss'], 'r-o', label='Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='r')
ax1.tick_params(axis='y', labelcolor='r')
ax1.set_xticks(history['epoch'])

ax2 = ax1.twinx()
l2, = ax2.plot(history['epoch'], history['accuracy'], 'b-s', label='Test Accuracy (%)')
ax2.set_ylabel('Accuracy (%)', color='b')
ax2.tick_params(axis='y', labelcolor='b')

lines = [l1, l2]
ax1.legend(lines, [l.get_label() for l in lines], loc='center right')

plt.title('TNModel Training Performance')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_plot.png')
print("Plot saved to 'training_plot.png'")
