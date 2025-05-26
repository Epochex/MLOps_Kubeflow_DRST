# ml/model.py
import torch, torch.nn as nn, torch.optim as optim

class MLP(nn.Module):
    def __init__(self, in_dim, h1=32, h2=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2),     nn.ReLU(),
            nn.Linear(h2, 1)
        )

    def forward(self, x):            # x: (N, in_dim)
        return self.net(x).squeeze(1)

def train_mlp(X_tr, y_tr, X_val, y_val,
              lr=1e-2, epochs=10, batch=4, seed=40):
    torch.manual_seed(seed)
    dev   = "cuda" if torch.cuda.is_available() else "cpu"
    mdl   = MLP(X_tr.shape[1]).to(dev)
    opt   = optim.Adam(mdl.parameters(), lr=lr)
    crit  = nn.MSELoss()

    ds  = torch.utils.data.TensorDataset(
            torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    dl  = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)

    for _ in range(epochs):
        mdl.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            loss   = crit(mdl(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()

    mdl.eval()
    with torch.no_grad():
        y_hat = mdl(torch.from_numpy(X_val).to(dev)).cpu().numpy()
    return mdl, y_hat
