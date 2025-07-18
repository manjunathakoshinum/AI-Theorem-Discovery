import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Dropout
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from torch.serialization import add_safe_globals
from torch_geometric.data.data import DataEdgeAttr
from tqdm import tqdm
import os

# ✅ Add safe global for PyTorch 2.6+
add_safe_globals([DataEdgeAttr])

# ✅ GATv2 model with improvements
class GATv2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2, dropout=0.3):
        super().__init__()
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.bn1 = BatchNorm1d(hidden_channels * heads)
        self.gat2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, dropout=dropout)
        self.dropout = Dropout(dropout)
        self.res_connection = Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        x_in = x
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)

        # Residual connection (project input to output size)
        x_res = self.res_connection(x_in)
        return x + x_res

# ✅ Label smoothing
def smooth_labels(labels, num_classes, smoothing=0.1):
    confidence = 1.0 - smoothing
    smoothed = torch.full((labels.size(0), num_classes), smoothing / (num_classes - 1)).to(labels.device)
    smoothed.scatter_(1, labels.unsqueeze(1), confidence)
    return smoothed

# ✅ Accuracy metric
def accuracy(preds, labels):
    return (preds.argmax(dim=1) == labels).sum().item() / labels.size(0)

# ✅ Safe graph loading for PyTorch 2.6+
def load_graph(path):
    try:
        return torch.load(path, weights_only=False)
    except Exception as e:
        print(f"❌ Error loading graph: {e}")
        raise

def main():
    input_path = 'saved_graphs/graph_data_masked.pt'
    print(f"📥 Loading masked graph from: {input_path}")
    data = load_graph(input_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    in_dim = data.x.size(-1)
    out_dim = int(data.y.max().item()) + 1

    train_idx, temp_idx = train_test_split(torch.arange(data.num_nodes), test_size=0.4, stratify=data.y.cpu())
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=data.y[temp_idx].cpu())

    model = GATv2(in_channels=in_dim, hidden_channels=32, out_channels=out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_val_acc = 0
    best_test_acc = 0
    best_model_state = None
    patience = 30
    trigger = 0

    print("🚀 Starting training...\n")
    for epoch in tqdm(range(1, 201)):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        smoothed = smooth_labels(data.y[train_idx], out_dim)
        loss = F.kl_div(F.log_softmax(out[train_idx], dim=1), smoothed, reduction='batchmean')
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index)
            val_acc = accuracy(pred[val_idx], data.y[val_idx])
            test_acc = accuracy(pred[test_idx], data.y[test_idx])

        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_model_state = model.state_dict()
            trigger = 0
        else:
            trigger += 1
            if trigger >= patience:
                print(f"\n⏹️ Early stopping at epoch {epoch} (best at {epoch - trigger}) due to no improvement.\n")
                break

    print("✅ Training complete.")
    print(f"📈 Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"🏁 Final Test Accuracy (Best Model): {best_test_acc:.4f}")

    if best_model_state:
        torch.save(best_model_state, "saved_graphs/best_gat_model.pt")
        print("📦 Best model saved as: best_gat_model.pt")

if __name__ == "__main__":
    main()
