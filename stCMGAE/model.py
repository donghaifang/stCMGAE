import copy
from functools import partial
import torch.nn.functional as F
import torch
from torch import nn
from torch_geometric.nn import (
    DeepGraphInfomax,
    TransformerConv,
    LayerNorm,
    Linear,
    GCNConv,
    SAGEConv,
    GATConv,
    GINConv,
    GATv2Conv,
    global_add_pool,
    global_mean_pool,
    global_max_pool
)

try:
    import torch_cluster  # noqa

    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import to_undirected, sort_edge_index
from torch_geometric.utils import add_self_loops, negative_sampling, degree
from torch_geometric.nn.inits import reset, uniform

def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, bn=True, dropout_rate=.1, act="prelu", bias=True):
        super().__init__()
        bn = nn.BatchNorm1d if bn else nn.Identity
        self.conv1 = GCNConv(in_channels=input_dim, out_channels=hidden_dim, heads=1, dropout=dropout_rate, concat=False, bias=bias)
        self.bn1 = bn(hidden_dim * 1)
        self.conv2 = GCNConv(in_channels=hidden_dim, out_channels=latent_dim, heads=1, dropout=dropout_rate, concat=False, bias=bias)
        self.bn2 = bn(latent_dim * 1)
        self.activation = create_activation(act)

    def forward(self, x, edge_index):
        h = self.activation(self.bn2(self.conv2(self.activation(self.bn1(self.conv1(x, edge_index))), edge_index)))
        return h

class FeatureDecoder(nn.Module):
    def __init__(self, latent_dim,  output_dim, dropout_rate=.1, act="prelu", bias=True):
        super().__init__()
        self.conv1 = GCNConv(in_channels=latent_dim, out_channels=output_dim, heads=1, dropout=dropout_rate, concat=False, bias=bias)
        self.activation = create_activation(act)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        return h

class stCMGAE_model(nn.Module):
    def __init__(self, features_dims, bn=False, att_dropout_rate=.2, use_token=True, alpha=2, feat_mask_rate=0.3, momentum=0.86):
        super().__init__()
        [input_dim, hidden_dim, latent_dim, output_dim] = features_dims
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, bn=bn, dropout_rate=att_dropout_rate, act="prelu", bias=True)
        self.teacher_encoder = copy.deepcopy(self.encoder)
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False
            p.detach_()

        self.projector = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.PReLU(),
            nn.Linear(256, latent_dim),
        )
        self.teacher_projector = copy.deepcopy(self.projector)
        for p in self.teacher_projector.parameters():
            p.requires_grad = False
            p.detach_()

        self.weight = nn.Parameter(torch.empty(latent_dim, latent_dim))
        uniform(latent_dim, self.weight)
        self.summary = lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0))
        self._momentum = momentum

        self.use_token = use_token
        if self.use_token:
            self.pos_enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))
            self.neg_enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))
        self.encoder_to_decoder = nn.Linear(latent_dim, latent_dim, bias=False)
        nn.init.xavier_uniform_(self.encoder_to_decoder.weight)
        self.feat_deocder = FeatureDecoder(latent_dim,  output_dim, dropout_rate=att_dropout_rate, act="prelu", bias=True)
        self.feat_loss = self.setup_loss_fn("sce", alpha)
        self.feat_mask_rate = feat_mask_rate

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        use_pos_x, mask_nodes, keep_nodes = self.mask_feature(x, self.feat_mask_rate)
        use_neg_x = self.corrupt_feature(x, mask_nodes, keep_nodes)

        rep_pos_x = self.encoder(use_pos_x, edge_index)
        rep_neg_x = self.teacher_encoder(use_neg_x, edge_index)

        pos_z = self.projector(rep_pos_x[mask_nodes])
        neg_z = self.teacher_projector(rep_neg_x[mask_nodes])

        s = self.summary(pos_z)
        dgi_loss = self.dgi_loss(pos_z, neg_z, summary=s)

        # remasking feats
        rec_pos_x = self.encoder_to_decoder(rep_pos_x)
        rec_pos_x[mask_nodes] = 0
        rec_pos_x = self.feat_deocder(rec_pos_x, edge_index)
        feat_loss = self.feat_loss(x[mask_nodes], rec_pos_x[mask_nodes])

        self.ema_update()
        return feat_loss, dgi_loss

    def ema_update(self):
        def update(student, teacher):
            with torch.no_grad():
            # m = momentum_schedule[it]  # momentum parameter
                m = self._momentum
                for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        update(self.encoder, self.teacher_encoder)
        update(self.projector, self.teacher_projector)

    def setup_loss_fn(self, loss_fn, alpha_l=2):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(self.sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def sce_loss(self, x, y, alpha):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
        loss = loss.mean()
        return loss

    def discriminate(self, z, summary, sigmoid=True):
        summary = summary.t() if summary.dim() > 1 else summary
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    def dgi_loss(self, pos_z, neg_z, summary):
        pos_loss = -torch.log(self.discriminate(pos_z, summary, sigmoid=True) + 1e-15).mean()
        neg_loss = -torch.log(1 - self.discriminate(neg_z, summary, sigmoid=True) + 1e-15).mean()
        return pos_loss + neg_loss

    def mask_feature(self, x, feat_mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        # random masking
        num_mask_nodes = int(feat_mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]
        out_x = x.clone()
        if self.use_token:
            out_x[mask_nodes] += self.pos_enc_mask_token
        else:
            out_x[mask_nodes] = 0.0
        return out_x, mask_nodes, keep_nodes

    def corrupt_feature(self, x, mask_nodes, keep_nodes):
        tmp = torch.zeros(x.shape).to(x.device)
        tmp[keep_nodes] = x[keep_nodes][torch.randperm(x[keep_nodes].shape[0])]
        if self.use_token:
            tmp[mask_nodes] += self.neg_enc_mask_token
        else:
            tmp[mask_nodes] = 0.0
        return tmp

    @torch.no_grad()
    def embed(self, data):
        x = data.x
        edge_index = data.edge_index
        h = self.encoder(x, edge_index)
        return h

    @torch.no_grad()
    def recon(self, data):
        x = data.x
        edge_index = data.edge_index
        h = self.encoder(x, edge_index)
        rec = self.encoder_to_decoder(h)
        rec = self.feat_deocder(rec, edge_index)
        return h, rec

