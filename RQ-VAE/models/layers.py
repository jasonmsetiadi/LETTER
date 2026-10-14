import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None



class MLPLayers(nn.Module):

    def __init__(
        self, layers, dropout=0.0, activation="relu", bn=False
    ):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None and idx != (len(self.layers)-2):
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.apply(self.init_weights)

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


from .gpt import GPT, norm


class StopPredictor(nn.Module):
    """
    Phase 2 learned stopping decision head using Causal Residual Transformer & Hazard Modeling.
    Takes the sequence of progressive residual states [h_0, h_1, ..., h_{T-1}] across quantization
    steps, processes them causally with a Transformer, and predicts hazard/stopping probabilities.
    """
    def __init__(self, in_dim, max_depth, hidden_dim=128, dropout=0.0):
        super(StopPredictor, self).__init__()
        self.in_dim = in_dim
        self.max_depth = max_depth
        self.hidden_dim = hidden_dim

        if in_dim != hidden_dim:
            self.input_proj = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
        else:
            self.input_proj = nn.Identity()

        num_heads = max(1, hidden_dim // 64)
        self.stop_transformer = GPT(
            num_layers=1,
            num_heads=num_heads,
            model_dim=hidden_dim,
            max_seq_len=max_depth,
            causal=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.stop_predictor = nn.Linear(hidden_dim, 1)
        self.stop_bias = nn.Parameter(torch.zeros(max_depth - 1))
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    @staticmethod
    def _process_stop_logits(stop_logits, return_survival_logits=False):
        log_cont = F.logsigmoid(-stop_logits)
        S = torch.cumsum(log_cont, dim=-1)
        length_logits = torch.cat([S - log_cont + F.logsigmoid(stop_logits), S[..., -1:]], dim=-1)
        if return_survival_logits:
            survival_logits = torch.cat([S.new_zeros(S.shape[:-1] + (1,)), S], dim=-1)
            return length_logits, survival_logits
        return length_logits

    def forward(self, residuals, length_mode="argmax"):
        """
        residuals:
          - Tensor [B, T, in_dim] containing progressive residual states h_0 .. h_{T-1}
          - OR Tensor [B, in_dim] (broadcasted to sequence for backward compatibility)
        """
        if residuals.ndim == 2:
            residuals = residuals.unsqueeze(1).repeat(1, self.max_depth, 1)

        B, T, _ = residuals.shape
        h = self.input_proj(residuals)
        states = self.stop_transformer(h)

        stop_logits = self.stop_predictor(self.dropout(states[:, 1:])).squeeze(-1) + self.stop_bias[None, :]

        length_logits, survival_logits = self._process_stop_logits(
            stop_logits.to(torch.float32), return_survival_logits=True
        )
        stop_probs = length_logits.softmax(dim=-1)
        survival_probs = survival_logits.exp().clamp(min=0.0, max=1.0)

        if length_mode == "argmax":
            pred_lengths = length_logits.argmax(dim=-1) + 1
        elif length_mode == "mean":
            depths = torch.arange(1, T + 1, device=residuals.device, dtype=residuals.dtype)
            E_L = (stop_probs * depths[None, :]).sum(dim=-1)
            pred_lengths = E_L.round().clamp(1, T).to(torch.long)
        elif length_mode == "median":
            cdf = stop_probs.cumsum(dim=-1)
            pred_lengths = (cdf >= 0.5).to(torch.long).argmax(dim=-1) + 1
        else:
            pred_lengths = length_logits.argmax(dim=-1) + 1

        return length_logits, stop_probs, survival_probs, pred_lengths


def activation_layer(activation_name="relu", emb_dim=None):

    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == "sigmoid":
            activation = nn.Sigmoid()
        elif activation_name.lower() == "tanh":
            activation = nn.Tanh()
        elif activation_name.lower() == "relu":
            activation = nn.ReLU()
        elif activation_name.lower() == "leakyrelu":
            activation = nn.LeakyReLU()
        elif activation_name.lower() == "none":
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError(
            "activation function {} is not implemented".format(activation_name)
        )

    return activation

def kmeans(
    samples,
    num_clusters,
    num_iters = 10,
):
    B, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device
    x = samples.cpu().detach().numpy()

    cluster = KMeans(n_clusters = num_clusters, max_iter = num_iters).fit(x)

    centers = cluster.cluster_centers_
    tensor_centers = torch.from_numpy(centers).to(device)

    return tensor_centers


@torch.no_grad()
def sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations):
    Q = torch.exp(-distances / epsilon)

    B = Q.shape[0] # number of samples to assign
    K = Q.shape[1] # how many centroids per block (usually set to 256)

    # make the matrix sums to 1
    sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    Q /= sum_Q
    # print(Q.sum())
    for it in range(sinkhorn_iterations):

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B

        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= K


    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q