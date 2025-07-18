import torch
from torch.nn import CrossEntropyLoss
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import os

torch.manual_seed(42)

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2, dropout=0.6):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x

def load_graph_safely(filepath):
    print(f"📥 Loading masked graph from: {filepath}")
    return torch.load(filepath, weights_only=False)  # Do NOT use weights_only=True for full .pt

def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out[mask].argmax(dim=1)
        acc = accuracy_score(data.y[mask].cpu(), pred.cpu())
    return acc

def main():
    input_file = "saved_graphs/graph_data_masked.pt"
    data = load_graph_safely(input_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    model = GAT(
        in_channels=data.x.size(1),
        hidden_channels=64,
        out_channels=len(torch.unique(data.y)),
        heads=4
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, 201):
        loss = train(model, data, optimizer, criterion)
        val_acc = evaluate(model, data, data.val_mask)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_gat_model.pt")

        if epoch % 10 == 0 or epoch == 1:
            test_acc = evaluate(model, data, data.test_mask)
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

    print("\n✅ Training complete. Best validation accuracy:", best_val_acc)
    print("📦 Best model saved as: best_gat_model.pt")

if __name__ == "__main__":
    main()
