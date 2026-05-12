import math
import torch
import torch.nn as nn

# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/optimal_transport.py
def log_otp_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    r"""Sinkhorn matrix scaling algorithm for Differentiable Optimal Transport problem.
    This function solves the optimization problem and returns the OT matrix for the given parameters.
    Args:
        log_a : torch.Tensor
            Source weights
        log_b : torch.Tensor
            Target weights
        M : torch.Tensor
            metric cost matrix
        num_iters : int, default=100
            The number of iterations.
        reg : float, default=1.0
            regularization value
    """
    M = M / reg  # regularization

    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()

    return M + u.unsqueeze(2) + v.unsqueeze(1)

# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/superglue.py
def get_matching_probs(S, dustbin_score = 1.0, num_iters=3, reg=1.0):
    """sinkhorn"""
    batch_size, m, n = S.size()
    # augment scores matrix
    # Dùng S.new_empty thay cho torch.empty(..., device=S.device) để tránh tracer 'baking' hằng số cpu
    S_aug = S.new_empty(batch_size, m + 1, n + 1)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :n] = dustbin_score
    S_aug[:, :m, n] = dustbin_score
    S_aug[:, m, n] = dustbin_score

    # prepare normalized source and target log-weights
    # Thay vì dùng torch.tensor, dùng S.new_tensor hoặc math float
    norm = math.log(m + n)
    log_a = S.new_zeros(batch_size, m + 1)
    log_b = S.new_zeros(batch_size, n + 1)

    log_P = log_otp_solver(
        log_a,
        log_b,
        S_aug,
        num_iters=num_iters,
        reg=reg
    )
    # Norm là float, trừ bình thường (Broadcasting)
    return (log_P + norm)[:, :, :-1]


class SALAD(nn.Module):
    """
    This class represents the Sinkhorn Algorithm for Locally Aggregated Descriptors (SALAD) model.

    Attributes:
        num_channels (int): The number of channels of the inputs (d).
        num_clusters (int): The number of clusters in the model (m).
        cluster_dim (int): The number of channels of the clusters (l).
        token_dim (int): The dimension of the global scene token (g).
        dropout (float): The dropout rate.
    """
    def __init__(self,
            num_channels=1536,
            num_clusters=64,
            cluster_dim=128,
            token_dim=256,
            dropout=0.3,
        ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.num_clusters= num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        
        if dropout > 0:
            dropout = nn.Dropout(dropout)
        else:
            dropout = nn.Identity()

        # MLP for global scene token g
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim)
        )
        # MLP for local features f_i
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1)
        )
        # MLP for score matrix S
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )


        # Cho qua một encoder nhỏ để học mối quan hệ cross-image
        self.place_encoder = nn.Sequential(
            nn.Conv1d(cluster_dim, cluster_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(cluster_dim, cluster_dim, kernel_size=1)
        )

        # Dustbin parameter z
        self.dust_bin = nn.Parameter(torch.tensor(1.))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model= cluster_dim * num_clusters,              # embedding dimension
            nhead=8,                 # số "đầu" attention song song
            dim_feedforward=8192,     # độ rộng MLP bên trong
            activation="gelu",        # hàm kích hoạt cho FFN
            dropout=0.1,              # tỉ lệ dropout
            batch_first=False         # input shape [seq_len, batch, dim]
        )

        self.cross_image_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)



    def forward(self, x):
        """
        x (tuple): A tuple containing two elements, f and t. 
            (torch.Tensor): The feature tensors (t_i) [B, C, H // 14, W // 14].
            (torch.Tensor): The token tensor (t_{n+1}) [B, C].

        Returns:
            f (torch.Tensor): The global descriptor [B, m*l + g]
        """
        x, t = x # Extract features and token

        # x [ B, C , H, W ] 
        f = self.cluster_features(x).flatten(2) # [batch , cluster_dim, num_patch]
        p = self.score(x).flatten(2)
        t = self.token_features(t)

        # Sinkhorn algorithm
        p = get_matching_probs(p, self.dust_bin, 3)
        p = torch.exp(p)
        # Normalize to maintain mass
        p = p[:, :-1, :] # [batch , num_cluster, num_patch]


        # [B, 1, num_clusters, num_patches] , [B, cluster_dim, num_clusters, num_patches] 
        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1) 

        # [B, cluster_dim, 1, num_patches]  [B, cluster_dim, num_clusters, num_patches] 
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)

        q = f * p                                  # [B, cluster_dim, num_clusters, num_patches]
        s = q.sum(dim=-1)    # [B, cluster_dim, num_clusters] 


        # -------------------------------
        # KHAI THÁC TÍNH CHẤT CHUNG GIỮA 4 ẢNH TRONG CÙNG MỘT PLACE
        # -------------------------------

        B, cluster_dim, num_clusters = s.shape
        img_per_place = 4
        num_places = B // img_per_place

        # Gom nhóm 4 ảnh theo từng place
        s_place = s.view(num_places, img_per_place, cluster_dim, num_clusters)
        
        # Flatten cluster features
        s_seq = s_place.flatten(2)  # [num_places, img_per_place, cluster_dim*num_clusters]

        # Permute để phù hợp với Transformer
        s_seq = s_seq.permute(1, 0, 2)  # [seq_len=img_per_place, batch=num_places, embed_dim]

        s_encoded = self.cross_image_encoder(s_seq) 

        # Chuyển ngược về [B, cluster_dim, num_clusters]
        s_encoded = s_encoded.permute(1, 0, 2)  # [num_places, img_per_place, embed_dim]
        s_encoded = s_encoded.contiguous().view(B, cluster_dim, num_clusters)

        s_encoded = nn.functional.normalize(s_encoded, p=2, dim=1)
        # ==== Thêm normalize chho s_encoded ở đây 
        s = s + s_encoded  # residual enhancement

        f = torch.cat([
            nn.functional.normalize(t, p=2, dim=-1),

            #  nhân từng phân tử với từng mẫu và cộng dồn theo chiều patches 
            #  [B, cluster_dim, num_clusters] 
            nn.functional.normalize(s, p=2, dim=1).flatten(1)
        ], dim=-1)

        return nn.functional.normalize(f, p=2, dim=-1)