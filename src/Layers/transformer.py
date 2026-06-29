import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class Multihead_Attention(nn.Module):
    """
    Multihead Self-Attention with optional stochastic latent token.
    """

    def __init__(self, dim: int, num_heads: int, seq_len: int, stochastic: bool):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.inv_sqrt_head_dim = 1.0 / math.sqrt(self.head_dim)
        self.stochastic = stochastic

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.z_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

        # causal mask for x -> x (query_len, key_len_without_noise)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf")),
            diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x):
        """
        x: (b, seq_len, dim)
        """

        b, seq_len, _ = x.shape

        # --------------------------------------------------
        # Build KV input (add noise token only if stochastic)
        # --------------------------------------------------
        if self.stochastic:
            z = torch.randn(b, self.dim, device=x.device)
            z_token = self.z_proj(z)                    # (b, dim)
            y = torch.cat([z_token.unsqueeze(1), x], dim=1)  # (b, seq_len+1, dim)
            key_len = seq_len + 1
        else:
            y = x
            key_len = seq_len

        # ----------------------
        # Projections
        # ----------------------
        q = self.q_proj(x).reshape(
            b, seq_len, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)                 # (b, h, seq_len, d)

        k = self.k_proj(y).reshape(
            b, key_len, self.num_heads, self.head_dim
        ).permute(0, 2, 3, 1)                 # (b, h, d, key_len)

        v = self.v_proj(y).reshape(
            b, key_len, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)                 # (b, h, key_len, d)

        # ----------------------
        # Attention logits
        # ----------------------
        att = (q @ k) * self.inv_sqrt_head_dim   # (b, h, seq_len, key_len)

        # ----------------------
        # Apply mask
        # ----------------------
        if self.stochastic:
            # allow attention to noise token (column 0)
            noise_col = torch.zeros(seq_len, 1, device=x.device)
            mask = torch.cat([noise_col, self.causal_mask], dim=1)
        else:
            mask = self.causal_mask

        att = att + mask.unsqueeze(0).unsqueeze(0)

        # ----------------------
        # Softmax + output
        # ----------------------
        a = self.softmax(att)
        o = a @ v

        o = self.o_proj(
            o.transpose(1, 2).reshape(b, seq_len, self.dim)
        )

        return o, a.mean(1)



class Attention_Block(torch.nn.Module):
    """
    Attention Block Module.
    """

    def __init__(self, dim: int, num_heads: int, seq_len: int, stochastic : bool):
        """
        Initialise Attention_Block module.

        Parameters:
        - dim       (int) : Dimension of input.
        - num_heads (int) : Number of attention heads.
        - seq_len   (int) : Length of input sequence.
        """

        super().__init__()
        self.stochastic = stochastic
        self.self_attention = Multihead_Attention(dim, num_heads, seq_len, stochastic = self.stochastic)

        self.ln_1 = torch.nn.LayerNorm(dim)
        self.ln_2 = torch.nn.LayerNorm(dim)
        self.ln_3 = torch.nn.LayerNorm(dim)
        self.ln_c = torch.nn.LayerNorm(dim)


        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 2, dim)
        )

    def forward(self, x) :#, t):

        # o, a = self.cross_attention(self.ln_1(x), self.ln_c(t))
        # x = x + o

        o, a = self.self_attention(self.ln_2(x))
        x = x + o

        o = self.mlp(self.ln_3(x))
        x = x + o

        return x, a

class TransformerModel(torch.nn.Module):
    """
    Transformer Module with Cross-Attention.
    """

    def __init__(self, args, model_eval=False):
        """
        Initialise Transformer module.

        Parameters:
        - args      (dict): Model arguments.
        - model_eval (bool): Evaluation mode flag.
        """
        super(TransformerModel, self).__init__()
        self.args = args
        print(f"Stochastic Transformer {self.args['nattblocks']} attention blocks, {self.args['hidden_dim']} hidden dimensions, {self.args['nheads']} head(s)")

        self.device = self.args["device"]
        self.seq_len = self.args["seq_len"]
        self.num_heads = self.args["nheads"]
        self.nattblocks = self.args['nattblocks']
        self.hidden_dim = self.args["num_obs"]
        self.model_eval = model_eval
        self.log_var_up = self.args['bound_sup']
        self.log_var_down = self.args['bound_inf']

        self.mu = nn.Linear(2*self.hidden_dim, self.hidden_dim)
        self.log_var =  nn.Linear(2*self.hidden_dim, self.hidden_dim)

        position = torch.arange(0, self.seq_len).unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, self.hidden_dim, 2) * (-math.log(10000.0) / self.hidden_dim))
        TE = torch.zeros(self.seq_len, self.hidden_dim)
        TE[:, 0::2] = torch.sin(position * div_term)
        TE[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('TE', TE)

        self.att_blocks = torch.nn.ModuleList([
            Attention_Block(
                self.hidden_dim,
                self.num_heads,
                self.seq_len,
                stochastic=False 
            )
            for i in range(self.nattblocks)
        ])

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        )

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)

        return mu + eps * std


    def forward(self, z) :

        z = z + self.TE

        for i, att_block in enumerate(self.att_blocks):
            z, a = att_block(z)

        z = self.mlp(z[:, -1])
        mu = self.mu(z)

        log_var = self.log_var(z)
        log_var = self.log_var_down + torch.nn.functional.softplus(log_var - self.log_var_down)  # smooth lower bound
        log_var = self.log_var_up - torch.nn.functional.softplus(self.log_var_up - log_var)     # smooth upper bound
        output = self.reparameterize(mu, log_var)
        

        return mu, log_var, output