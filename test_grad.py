import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import quimb.tensor as qt
from tqdm import tqdm

# --- 1. The Model (Kept as requested, with minor list fix in forward) ---
class TNModel(nn.Module):
    def __init__(self, tn_pixels, tn_patches, output_dims, input_dims):
        super().__init__()
        
        # Pack parameters
        # tn_pixels and tn_patches must be TensorNetwork objects, not lists
        params_pixel, self.skeleton_pixels = qt.pack(tn_pixels)
        params_patches, self.skeleton_patches = qt.pack(tn_patches)

        # Initialize Parameters
        self.torch_params_pixels = nn.ParameterDict({
            str(i): nn.Parameter(initial) for i, initial in params_pixel.items()
        })
        self.torch_params_patches = nn.ParameterDict({
            str(i): nn.Parameter(initial) for i, initial in params_patches.items()
        })
        
        self.input_dims = input_dims
        self.output_dims = output_dims

    def forward(self, x):
        # Unpack Pixels
        params_pixels = {int(i): p for i, p in self.torch_params_pixels.items()}
        pixels = qt.unpack(params_pixels, self.skeleton_pixels)
        
        # Unpack Patches
        params_patches = {int(i): p for i, p in self.torch_params_patches.items()}
        patches = qt.unpack(params_patches, self.skeleton_patches)
        
        # Create Input Nodes
        input_nodes_list = self.construct_nodes(x)
        
        # Convert list to TN so we can use '&'
        input_tn = qt.TensorNetwork(input_nodes_list)

        # Contract
        tn = pixels & patches & input_tn
        out = tn.contract(output_inds=self.output_dims, optimize='auto-hq')
        
        return out.data

    def construct_nodes(self, x):
        input_nodes = []
        for idx, i in enumerate(self.input_dims):
            a = qt.Tensor(x[:, idx], inds=["s", f"{i}_pixels", f"{i}_patches"], tags=f"Input_{i}")
            input_nodes.append(a)
        return input_nodes

# --- 2. Data Preparation ---

BATCH_SIZE = 1000
BLOCKS = 3
DIM_P = 16       
DIM_PT = 16 
TARGET_SIZE = BLOCKS * DIM_P * DIM_PT # 768

def reshape_and_pad(x):
    x = x.view(-1)[:TARGET_SIZE] 
    x = x.view(BLOCKS, DIM_P, DIM_PT)
    x = torch.nn.functional.pad(x, (0, 1, 0, 1)) 
    x[..., -1, -1] = 1.0
    return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(reshape_and_pad)
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. Manual Tensor Construction ---

bond_dim = 12
phys_dim_p = DIM_P + 1   # 17
phys_dim_pt = DIM_PT + 1 # 17

def init_data(*shape):
    return torch.randn(*shape) * 0.1

# === A. Define Pixels Tensors ===
pixels_mps_list = [
    qt.Tensor(data=init_data(1, phys_dim_p, bond_dim), inds=["_null_p_l", "0_pixels", "bond_p_01"], tags=["PIXELS", "0"]),
    qt.Tensor(data=init_data(bond_dim, phys_dim_p, bond_dim, 10), inds=["bond_p_01", "1_pixels", "bond_p_12", "class_out"], tags=["PIXELS", "1", "OUTPUT"]),
    qt.Tensor(data=init_data(bond_dim, phys_dim_p, 1), inds=["bond_p_12", "2_pixels", "_null_p_r"], tags=["PIXELS", "2"])
]

# === B. Define Patches Tensors ===
patches_mps_list = [
    qt.Tensor(data=init_data(1, phys_dim_pt, bond_dim), inds=["_null_pt_l", "0_patches", "bond_pt_01"], tags=["PATCHES", "0"]),
    qt.Tensor(data=init_data(bond_dim, phys_dim_pt, bond_dim), inds=["bond_pt_01", "1_patches", "bond_pt_12"], tags=["PATCHES", "1"]),
    qt.Tensor(data=init_data(bond_dim, phys_dim_pt, 1), inds=["bond_pt_12", "2_patches", "_null_pt_r"], tags=["PATCHES", "2"])
]

# --- 4. Initialize & Train ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_labels = ["0", "1", "2"] 

# !!! FIX HERE: Wrap the lists in qt.TensorNetwork() !!!
model = TNModel(
    tn_pixels=qt.TensorNetwork(pixels_mps_list), 
    tn_patches=qt.TensorNetwork(patches_mps_list), 
    output_dims=["s", "class_out"],
    input_dims=input_labels
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

import matplotlib.pyplot as plt

# ... (Previous code: Imports, Model Class, Data Prep, Model Init) ...

# --- 5. Training with History & Plotting ---

# Store history
history = {
    'epoch': [],
    'loss': [],
    'accuracy': []
}

epochs = 10
print(f"Model initialized on {device}. Starting training...")
# Define Evaluate
def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # output is a raw tensor now
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
    
    # Calculate average loss for the epoch
    avg_loss = running_loss / len(train_loader)
    
    # Evaluate
    acc = evaluate(test_loader)
    print(f"Epoch {epoch+1} Test Accuracy: {acc*100:.2f}%")
    
    # Store metrics
    history['epoch'].append(epoch + 1)
    history['loss'].append(avg_loss)
    history['accuracy'].append(acc * 100)

# --- 6. Visualization ---

plt.figure(figsize=(10, 5))

# Plot Loss
ax1 = plt.gca()
l1, = ax1.plot(history['epoch'], history['loss'], 'r-o', label='Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='r')
ax1.tick_params(axis='y', labelcolor='r')
ax1.set_xticks(history['epoch']) # Ensure integer ticks for epochs

# Create a twin axis for Accuracy
ax2 = ax1.twinx()
l2, = ax2.plot(history['epoch'], history['accuracy'], 'b-s', label='Test Accuracy (%)')
ax2.set_ylabel('Accuracy (%)', color='b')
ax2.tick_params(axis='y', labelcolor='b')

# Add legend
lines = [l1, l2]
ax1.legend(lines, [l.get_label() for l in lines], loc='center right')

plt.title('TNModel Training Performance')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save and Show
plt.savefig('training_plot.png')
print("Plot saved to 'training_plot.png'")
plt.show()
