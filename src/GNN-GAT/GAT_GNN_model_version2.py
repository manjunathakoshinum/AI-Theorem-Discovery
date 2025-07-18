import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch_geometric.nn import GATConv
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
import argparse
from tqdm import trange
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
    return torch.load(filepath, weights_only=False)

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

def normalize_features(data, device):
    scaler = StandardScaler()
    data.x = torch.tensor(scaler.fit_transform(data.x.cpu()), dtype=torch.float).to(device)
    return data

def main(args):
    input_file = args.input_file
    data = load_graph_safely(input_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    data.y = data.y.long()  # Ensure labels are correct type
    data = normalize_features(data, device)  # Optional: Normalize features

    model = GAT(
        in_channels=data.x.size(1),
        hidden_channels=args.hidden,
        out_channels=len(torch.unique(data.y)),
        heads=args.heads,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = CrossEntropyLoss()

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in trange(1, args.epochs + 1):
        loss = train(model, data, optimizer, criterion)
        val_acc = evaluate(model, data, data.val_mask)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_args': {
                    'in_channels': data.x.size(1),
                    'hidden_channels': args.hidden,
                    'out_channels': len(torch.unique(data.y)),
                    'heads': args.heads,
                    'dropout': args.dropout
                }
            }, args.model_path)
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            test_acc = evaluate(model, data, data.test_mask)
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

        if patience_counter >= args.patience:
            print(f"⏹️ Early stopping at epoch {epoch} due to no improvement.")
            break

    print("\n✅ Training complete. Best validation accuracy:", best_val_acc)
    print(f"📦 Best model saved as: {args.model_path}")

    # Load best model and report final test accuracy
    checkpoint = torch.load(args.model_path)
    best_model = GAT(**checkpoint['model_args']).to(device)
    best_model.load_state_dict(checkpoint['model_state_dict'])
    final_test_acc = evaluate(best_model, data, data.test_mask)
    print(f"🏁 Final Test Accuracy (Best Model): {final_test_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GAT on theorem graph.")
    parser.add_argument("--input_file", type=str, default="saved_graphs/graph_data_masked.pt")
    parser.add_argument("--model_path", type=str, default="best_gat_model.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    main(args)
