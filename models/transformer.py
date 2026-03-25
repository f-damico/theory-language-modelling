import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcn import MLP


def _scale_parameter_(param, init_scale):
    if init_scale != 1.0:
        with torch.no_grad():
            param.mul_(init_scale)


class _TransformerNormMixin:
    @staticmethod
    def _weight_to_matrix(weight: torch.Tensor) -> torch.Tensor:
        weight64 = weight.detach().to(torch.float64)
        if weight64.ndim > 2:
            return weight64.view(weight64.size(0), -1)
        return weight64

    @classmethod
    def _spectral_norm(cls, weight: torch.Tensor) -> torch.Tensor:
        w = cls._weight_to_matrix(weight)
        return torch.linalg.svdvals(w)[0]

    @classmethod
    def _two_one_norm_transpose(cls, weight: torch.Tensor) -> torch.Tensor:
        w = cls._weight_to_matrix(weight)
        return torch.norm(w.T, p=2, dim=0).sum()

    def _all_weight_matrices_no_qk(self):
        for weight in self._all_weight_matrices():
            yield weight

    def _norm_from_weight_iterable(self, weights, *, device, dtype_out) -> torch.Tensor:
        weights = list(weights)
        if len(weights) == 0:
            return torch.tensor(0.0, device=device, dtype=dtype_out)

        specs = [self._spectral_norm(weight) for weight in weights]
        two1s = [self._two_one_norm_transpose(weight) for weight in weights]

        prod_spec = torch.prod(torch.stack(specs))
        correction = torch.stack([
            (t ** (2.0 / 3.0)) / (s ** (2.0 / 3.0) + 1e-12)
            for t, s in zip(two1s, specs)
        ]).sum()
        return (prod_spec * (correction ** 1.5)).to(dtype=dtype_out)

    @torch.no_grad()
    def compute_model_norm(self) -> torch.Tensor:
        """
        Spectral complexity norm with all selected transformer weights.
        """
        params = list(self.parameters())
        if len(params) == 0:
            return torch.tensor(0.0)

        device = params[0].device
        dtype_out = params[0].dtype
        return self._norm_from_weight_iterable(
            self._all_weight_matrices(),
            device=device,
            dtype_out=dtype_out,
        )

    @torch.no_grad()
    def compute_model_norm_no_qk(self) -> torch.Tensor:
        """
        Spectral complexity norm after removing query and key matrices from the
        list of weights entering the Bartlett-style proxy.
        """
        params = list(self.parameters())
        if len(params) == 0:
            return torch.tensor(0.0)

        device = params[0].device
        dtype_out = params[0].dtype
        return self._norm_from_weight_iterable(
            self._all_weight_matrices_no_qk(),
            device=device,
            dtype_out=dtype_out,
        )

    @torch.no_grad()
    def compute_l2_norm(self) -> torch.Tensor:
        """
        Global L2 norm of the trainable weight matrices used in the transformer.
        This is the Frobenius norm over the selected weights.
        """
        params = list(self.parameters())
        if len(params) == 0:
            return torch.tensor(0.0)

        device = params[0].device
        dtype_out = params[0].dtype

        total_sq = torch.tensor(0.0, dtype=torch.float64, device=device)
        for weight in self._all_weight_matrices():
            total_sq += torch.sum(weight.detach().to(torch.float64) ** 2)

        return torch.sqrt(total_sq).to(dtype=dtype_out)


class MultiHeadAttention(nn.Module):
    """
    Multiple Attention Heads.

    Args:
        input_dim: The dimension of input tokens.
        num_heads: The number of heads.
        head_dim: The dimension of each head.
        out_dim: The dimension of output tokens.
        init_scale: Multiplicative factor applied to the initial random weights.
    """
    def __init__(
        self, input_dim, num_heads, head_dim, out_dim, init_scale=1.0
    ):
        super().__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = head_dim * num_heads
        self.init_scale = init_scale

        self.key = nn.Parameter(
            torch.randn(self.inner_dim, self.input_dim)
        )
        self.query = nn.Parameter(
            torch.randn(self.inner_dim, self.input_dim)
        )
        self.value = nn.Parameter(
            torch.randn(self.inner_dim, self.input_dim)
        )
        self.projection = nn.Parameter(
            torch.randn(self.out_dim, self.inner_dim)
        )

        _scale_parameter_(self.key, self.init_scale)
        _scale_parameter_(self.query, self.init_scale)
        _scale_parameter_(self.value, self.init_scale)
        _scale_parameter_(self.projection, self.init_scale)

    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, input_size, input_dim).

        Returns:
            The output of a multi-head attention layer,
            of size (batch_size, input_size, output_dim)
        """
        B,T,C = x.size()
        k = F.linear( x, self.key, bias=None).view(B, T, self.num_heads, self.head_dim).transpose(1,2) *C**-.5
        q = F.linear( x, self.query, bias=None).view(B, T, self.num_heads, self.head_dim).transpose(1,2) *C**-.5
        v = F.linear( x, self.value, bias=None).view(B, T, self.num_heads, self.head_dim).transpose(1,2) *C**-.5

        weight = q @ k.transpose(-2,-1) * self.head_dim**-.5
        weight = F.softmax(weight, dim=-1)

        out = (weight @ v).transpose(1,2).reshape(B,T,-1)
        out = F.linear( out, self.projection, bias=None) * self.projection.size(-1)**-.5

        return out


class AttentionBlock(nn.Module):
    def __init__(
        self, embedding_dim, num_heads, init_scale=1.0
    ):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding dim. must be multiple of num. heads"

        self.sa = MultiHeadAttention(
            input_dim=embedding_dim,
            num_heads=num_heads,
            head_dim=embedding_dim//num_heads,
            out_dim=embedding_dim,
            init_scale=init_scale,
        )

    def forward(self, x):
        x = self.sa(x)
        return x


class EncoderBlock(nn.Module):
    """
    One Decoder Block.

    Args:
        embedding_dim: The dimension of the tokens (kept constant past embedding).
        num_heads: The number of attention heads.
        ffwd_size: Size of the MLP is ffwd_size*embedding_dim.
        init_scale: Multiplicative factor applied to the initial random weights.
    """
    def __init__(
        self, embedding_dim, num_heads, ffwd_size=4, init_scale=1.0
    ):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding dim. must be multiple of num. heads"

        self.sa = MultiHeadAttention(
            input_dim=embedding_dim,
            num_heads=num_heads,
            head_dim=embedding_dim//num_heads,
            out_dim=embedding_dim,
            init_scale=init_scale,
        )
        self.ffwd = MLP(
            input_dim=embedding_dim,
            nn_dim=ffwd_size*embedding_dim,
            out_dim=embedding_dim,
            num_layers=1
        )
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

        for layer in self.ffwd.hidden:
            linear_layer = layer[0]
            if hasattr(linear_layer, "weight") and linear_layer.weight is not None:
                _scale_parameter_(linear_layer.weight, init_scale)
            if hasattr(linear_layer, "bias") and linear_layer.bias is not None:
                _scale_parameter_(linear_layer.bias, init_scale)

        _scale_parameter_(self.ffwd.readout, init_scale)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MLA(_TransformerNormMixin, nn.Module):
    """
    Multi-Layer Multi-Head Attention for last token prediction

    Args:
        vocab_size: The dimension of input tokens.
        block_size: The (maximal) number of input tokens.
        embedding_dim: The embedding dimension.
        num_heads: The number of attention heads.
        num_layers: The number of layers.
        init_scale: Multiplicative factor applied to the initial random weights.
    """
    def __init__(
        self, vocab_size, block_size, embedding_dim, num_heads, num_layers, init_scale=1.0
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.init_scale = init_scale

        self.token_embedding = nn.Parameter(
            torch.randn(self.embedding_dim, self.vocab_size)
        )
        _scale_parameter_(self.token_embedding, self.init_scale)

        self.position_embedding = nn.Embedding(self.block_size, self.embedding_dim)
        _scale_parameter_(self.position_embedding.weight, self.init_scale)

        self.blocks = nn.Sequential(
            *[
                AttentionBlock(
                    embedding_dim=self.embedding_dim,
                    num_heads=self.num_heads,
                    init_scale=self.init_scale,
                ) for _ in range(self.num_layers)
            ]
        )
        self.readout = nn.Parameter(
            torch.randn(self.vocab_size, self.embedding_dim)
        )
        _scale_parameter_(self.readout, self.init_scale)

    def _all_weight_matrices(self):
        yield self.token_embedding
        for block in self.blocks:
            yield block.sa.key
            yield block.sa.query
            yield block.sa.value
            yield block.sa.projection
        yield self.readout

    def _all_weight_matrices_no_qk(self):
        yield self.token_embedding
        for block in self.blocks:
            yield block.sa.value
            yield block.sa.projection
        yield self.readout

    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, seq_len, vocab_size).

        Returns:
            Output of multilayer self-attention, tensor of size (batch_size, seq_len, vocab_size)
        """
        B,T,C = x.size()
        token_emb = F.linear( x, self.token_embedding, bias=None) *C**-.5
        pos_emb = self.position_embedding(torch.arange(T, device=x.device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        logits = F.linear( x[:,-1,:], self.readout, bias=None) * self.readout.size(-1)**-.5

        return logits


class BERTuccia(_TransformerNormMixin, nn.Module):
    """
    Bert-like model for last token prediction

    Args:
        vocab_size: The dimension of input tokens.
        block_size: The (maximal) number of input tokens.
        embedding_dim: The embedding dimension.
        num_heads: The number of attention heads.
        num_layers: The number of layers.
        init_scale: Multiplicative factor applied to the initial random weights.
    """
    def __init__(
        self, vocab_size, block_size, embedding_dim, num_heads, num_layers, init_scale=1.0
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.init_scale = init_scale

        self.token_embedding = nn.Parameter(
            torch.randn(self.embedding_dim, self.vocab_size)
        )
        _scale_parameter_(self.token_embedding, self.init_scale)

        self.position_embedding = nn.Embedding(self.block_size, self.embedding_dim)
        _scale_parameter_(self.position_embedding.weight, self.init_scale)

        self.blocks = nn.Sequential(
            *[
                EncoderBlock(
                    embedding_dim=self.embedding_dim,
                    num_heads=self.num_heads,
                    init_scale=self.init_scale,
                ) for _ in range(self.num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(self.embedding_dim)
        self.readout = nn.Parameter(
            torch.randn(self.vocab_size, self.embedding_dim)
        )
        _scale_parameter_(self.readout, self.init_scale)

    def _all_weight_matrices(self):
        yield self.token_embedding
        for block in self.blocks:
            yield block.sa.key
            yield block.sa.query
            yield block.sa.value
            yield block.sa.projection
            for layer in block.ffwd.hidden:
                linear_layer = layer[0]
                yield linear_layer.weight
            yield block.ffwd.readout.T
        yield self.readout

    def _all_weight_matrices_no_qk(self):
        yield self.token_embedding
        for block in self.blocks:
            yield block.sa.value
            yield block.sa.projection
            for layer in block.ffwd.hidden:
                linear_layer = layer[0]
                yield linear_layer.weight
            yield block.ffwd.readout.T
        yield self.readout

    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, seq_len, vocab_size).

        Returns:
            Probability of the last token, tensor of size (batch_size, vocab_size)
        """
        B,T,C = x.size()
        token_emb = F.linear( x, self.token_embedding, bias=None) *C**-.5
        pos_emb = self.position_embedding(torch.arange(T, device=x.device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = F.linear( x[:,-1,:], self.readout, bias=None) * self.readout.size(-1)**-.5

        return logits
