import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def build_signed_graph(num_nodes=50, num_edges=120, feature_dim=16, num_classes=3, seed=42):
    """
    Construct a synthetic signed graph with node features and class labels.
    Signed graph: edges have positive (+1) or negative (-1) relation types.
    """
    g = torch.Generator().manual_seed(seed)
    # Random node features
    X = torch.randn(num_nodes, feature_dim, generator=g)
    
    # Random class assignments based on linear projection
    W_true = torch.randn(feature_dim, num_classes, generator=g)
    logits = X @ W_true
    y = torch.argmax(logits, dim=1)
    
    # Generate edges
    edge_list = []
    signs = []
    edges_set = set()
    
    # Ensure graph connectivity and homophily/heterophily based on signs
    attempts = 0
    while len(edge_list) < num_edges and attempts < 1000:
        attempts += 1
        u = torch.randint(0, num_nodes, (1,), generator=g).item()
        v = torch.randint(0, num_nodes, (1,), generator=g).item()
        if u == v or (u, v) in edges_set or (v, u) in edges_set:
            continue
        edges_set.add((u, v))
        edge_list.append((u, v))
        
        # If same class -> positive edge (+1), if different class -> negative edge (-1)
        if y[u] == y[v]:
            signs.append(1.0)
        else:
            signs.append(-1.0)
            
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    edge_signs = torch.tensor(signs, dtype=torch.float32)
    
    return X, y, edge_index, edge_signs

class SheafLaplacian(nn.Module):
    """
    Constructs normalized Sheaf Laplacian operator given restriction maps or edge signs.
    Sheaf Laplacian captures signed, asymmetric, and varying-dimensional relations (Section 3).
    When restriction maps are identity matrices, this reduces to standard graph Laplacian (Section 2.1).
    """
    def __init__(self, num_nodes, feature_dim, is_identity=False):
        super().__init__()
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.is_identity = is_identity

    def compute_sheaf_matrix(self, edge_index, edge_signs=None):
        """
        Build (N*d, N*d) Sheaf Laplacian matrix.
        For node u, v with edge e=(u,v) and sign s_e:
        Restriction maps F_{u->e} = I_d, F_{v->e} = s_e * I_d.
        """
        N = self.num_nodes
        d = self.feature_dim
        L_sheaf = torch.zeros(N * d, N * d)
        degree = torch.zeros(N * d)

        num_edges = edge_index.shape[1]
        for e in range(num_edges):
            u, v = edge_index[0, e].item(), edge_index[1, e].item()
            s_e = 1.0 if (self.is_identity or edge_signs is None) else edge_signs[e].item()

            # F_{u->e} = I, F_{v->e} = s_e * I
            # Diagonal contributions
            idx_u = slice(u * d, (u + 1) * d)
            idx_v = slice(v * d, (v + 1) * d)

            L_sheaf[idx_u, idx_u] += torch.eye(d)
            L_sheaf[idx_v, idx_v] += torch.eye(d)

            # Off-diagonal block: - F_{u->e}^T F_{v->e} = - s_e * I
            L_sheaf[idx_u, idx_v] -= s_e * torch.eye(d)
            L_sheaf[idx_v, idx_u] -= s_e * torch.eye(d)

            degree[idx_u] += 1.0
            degree[idx_v] += 1.0

        # Normalized Sheaf Laplacian: D^{-1/2} L D^{-1/2}
        deg_inv_sqrt = torch.rsqrt(torch.clamp(degree, min=1.0))
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        L_norm = D_inv_sqrt @ L_sheaf @ D_inv_sqrt

        # Sheaf Diffusion operator: P = I - 0.5 * L_norm
        P_sheaf = torch.eye(N * d) - 0.5 * L_norm
        return P_sheaf

    def forward(self, X, edge_index, edge_signs=None):
        """
        X: (N, d) feature tensor
        Returns: (N, d) diffused feature tensor
        """
        N, d = X.shape
        P_sheaf = self.compute_sheaf_matrix(edge_index, edge_signs)
        x_flat = X.reshape(N * d, 1)
        out_flat = P_sheaf @ x_flat
        return out_flat.reshape(N, d)

class SheafGCN(nn.Module):
    """
    Sheaf Neural Network for node classification (Section 3).
    Replaces standard graph Laplacian diffusion with Sheaf-Laplacian diffusion.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.sheaf_op1 = SheafLaplacian(num_nodes, in_dim, is_identity=False)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.sheaf_op2 = SheafLaplacian(num_nodes, hidden_dim, is_identity=False)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, X, edge_index, edge_signs=None):
        h = self.sheaf_op1(X, edge_index, edge_signs)
        h = F.relu(self.fc1(h))
        h = self.sheaf_op2(h, edge_index, edge_signs)
        out = self.fc2(h)
        return out

class KipfWellingGCN(nn.Module):
    """
    Standard Kipf & Welling GCN variant (Section 2.1 / baseline).
    Uses standard symmetric normalized graph Laplacian (ignoring signed edge structure).
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.sheaf_op1 = SheafLaplacian(num_nodes, in_dim, is_identity=True)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.sheaf_op2 = SheafLaplacian(num_nodes, hidden_dim, is_identity=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, X, edge_index, edge_signs=None):
        # Ignores edge_signs (treats all edges with identity restriction maps)
        h = self.sheaf_op1(X, edge_index, None)
        h = F.relu(self.fc1(h))
        h = self.sheaf_op2(h, edge_index, None)
        out = self.fc2(h)
        return out
