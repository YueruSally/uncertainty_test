"""Transformer and MLP actor-critic models with identical action semantics."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .config import OPERATORS
from .observations import CANDIDATE_FEATURES, GLOBAL_FEATURES, PARENT_FEATURES


TRANSFORMER_ACTOR_CONDITIONING = (
    "candidate-plus-operator-plus-global-before-encoder")


def _signed_log1p(value):
    return torch.sign(value) * torch.log1p(torch.abs(value))


def _masked_mean(values, padding_mask):
    valid = (~padding_mask).unsqueeze(-1).to(values.dtype)
    return (values * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


class _Heads(nn.Module):
    def __init__(self, hidden_dim, action_count, critic_contexts=3,
                 zero_actor=False):
        super().__init__()
        self.actor_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(action_count)])
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim * critic_contexts, hidden_dim),
            nn.Tanh(), nn.Linear(hidden_dim, 1))
        if zero_actor:
            for head in self.actor_heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def actor_logits(self, candidate_hidden, operator_id):
        all_logits = torch.stack(
            [head(candidate_hidden).squeeze(-1) for head in self.actor_heads], dim=1)
        index = operator_id[:, None, None].expand(-1, 1, candidate_hidden.shape[1])
        return all_logits.gather(1, index).squeeze(1)


def _apply_actor_policy(logits, batch, actor_policy, residual_logit_limit):
    if actor_policy == "direct":
        return logits
    if actor_policy != "base_residual":
        raise ValueError(f"unknown actor policy: {actor_policy}")
    prior = batch["action_prior"].to(logits.dtype)
    bounded_residual = residual_logit_limit * torch.tanh(
        logits / residual_logit_limit)
    return bounded_residual + torch.log(prior.clamp_min(1e-12))


class NetworkTransformerEncoder(nn.Module):
    """Encode node/edge attribute tokens without requiring a GNN."""

    def __init__(self, node_dim, edge_dim, hidden_dim, heads, layers, dropout):
        super().__init__()
        self.node_projection = nn.Sequential(
            nn.Linear(node_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.edge_projection = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.node_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.edge_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, batch):
        node_mask = batch["graph_node_padding_mask"]
        edge_mask = batch["graph_edge_padding_mask"]
        nodes = self.node_projection(batch["graph_nodes"]) + self.node_type
        edges = self.edge_projection(batch["graph_edges"]) + self.edge_type
        source = batch["graph_edge_index"][..., 0]
        destination = batch["graph_edge_index"][..., 1]
        hidden = nodes.shape[-1]
        source_hidden = torch.gather(
            nodes, 1, source.unsqueeze(-1).expand(-1, -1, hidden))
        destination_hidden = torch.gather(
            nodes, 1, destination.unsqueeze(-1).expand(-1, -1, hidden))
        edges = edges + source_hidden + destination_hidden
        tokens = torch.cat([nodes, edges], dim=1)
        padding_mask = torch.cat([node_mask, edge_mask], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        encoded_nodes = encoded[:, :nodes.shape[1]]
        encoded_edges = encoded[:, nodes.shape[1]:]
        return encoded_nodes, encoded_edges, _masked_mean(encoded, padding_mask)


class NetworkMeanEncoder(nn.Module):
    """Non-attention ablation over the same node and edge attributes."""

    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.node_projection = nn.Sequential(
            nn.Linear(node_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.edge_projection = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())

    def forward(self, batch):
        nodes = self.node_projection(batch["graph_nodes"])
        edges = self.edge_projection(batch["graph_edges"])
        source = batch["graph_edge_index"][..., 0]
        destination = batch["graph_edge_index"][..., 1]
        hidden = nodes.shape[-1]
        edges = edges + torch.gather(
            nodes, 1, source.unsqueeze(-1).expand(-1, -1, hidden))
        edges = edges + torch.gather(
            nodes, 1, destination.unsqueeze(-1).expand(-1, -1, hidden))
        tokens = torch.cat([nodes, edges], dim=1)
        padding_mask = torch.cat([
            batch["graph_node_padding_mask"], batch["graph_edge_padding_mask"]], dim=1)
        return nodes, edges, _masked_mean(tokens, padding_mask)


def _candidate_graph_context(node_hidden, batch):
    indices = batch["candidate_graph_node_index"]
    mask = batch["candidate_graph_node_padding_mask"]
    hidden = node_hidden.shape[-1]
    expanded_nodes = node_hidden.unsqueeze(1).expand(
        -1, indices.shape[1], -1, -1)
    gathered = torch.gather(
        expanded_nodes, 2,
        indices.unsqueeze(-1).expand(-1, -1, -1, hidden))
    valid = (~mask).unsqueeze(-1).to(gathered.dtype)
    return ((gathered * valid).sum(dim=2)
            / valid.sum(dim=2).clamp_min(1.0))


def _ordered_graph_tokens(node_hidden, edge_hidden, batch, type_embedding,
                          segment_embedding, position_projection):
    indices = batch["candidate_graph_token_index"]
    kinds = batch["candidate_graph_token_type"]
    segments = batch["candidate_graph_token_segment"].clamp(0, 1)
    count, length = indices.shape[1], indices.shape[2]
    hidden = node_hidden.shape[-1]
    node_source = node_hidden.unsqueeze(1).expand(-1, count, -1, -1)
    edge_source = edge_hidden.unsqueeze(1).expand(-1, count, -1, -1)
    node_tokens = torch.gather(
        node_source, 2, indices.clamp_max(node_hidden.shape[1] - 1)
        .unsqueeze(-1).expand(-1, -1, -1, hidden))
    edge_tokens = torch.gather(
        edge_source, 2, indices.clamp_max(edge_hidden.shape[1] - 1)
        .unsqueeze(-1).expand(-1, -1, -1, hidden))
    tokens = torch.where(kinds.unsqueeze(-1).bool(), edge_tokens, node_tokens)
    position = _signed_log1p(
        batch["candidate_graph_token_position"].unsqueeze(-1))
    tokens = (tokens + type_embedding(kinds) + segment_embedding(segments)
              + position_projection(position))
    return tokens.reshape(-1, length, hidden)


class OrderedCandidatePathEncoder(nn.Module):
    """Encode each candidate's ordered node-edge path without objective calls."""

    def __init__(self, hidden_dim, heads, dropout, use_attention=True):
        super().__init__()
        self.type_embedding = nn.Embedding(2, hidden_dim)
        self.segment_embedding = nn.Embedding(2, hidden_dim)
        self.position_projection = nn.Linear(1, hidden_dim)
        self.encoder = None
        if use_attention:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=heads,
                dim_feedforward=hidden_dim * 2, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, node_hidden, edge_hidden, batch):
        source = _ordered_graph_tokens(
            node_hidden, edge_hidden, batch, self.type_embedding,
            self.segment_embedding, self.position_projection)
        mask = batch["candidate_graph_token_padding_mask"].reshape(
            -1, source.shape[1])
        valid_rows = ~mask.all(dim=1)
        pooled = source.new_zeros(source.shape[0], source.shape[-1])
        if valid_rows.any():
            encoded = source[valid_rows]
            valid_mask = mask[valid_rows]
            if self.encoder is not None:
                encoded = self.encoder(
                    encoded, src_key_padding_mask=valid_mask)
            pooled[valid_rows] = _masked_mean(encoded, valid_mask)
        return pooled.reshape(
            batch["candidate_graph_token_index"].shape[0],
            batch["candidate_graph_token_index"].shape[1], -1)


class TransformerActorCritic(nn.Module):
    """Candidate self-attention followed by parent-candidate cross-attention."""

    architecture = "transformer"

    def __init__(self, hidden_dim=128, heads=4, layers=2, dropout=0.1,
                 parent_dim=len(PARENT_FEATURES),
                 candidate_dim=len(CANDIDATE_FEATURES),
                 global_dim=len(GLOBAL_FEATURES), action_count=len(OPERATORS),
                 graph_node_dim=None, graph_edge_dim=None,
                 actor_policy="direct", residual_logit_limit=2.0,
                 candidate_representation="legacy"):
        super().__init__()
        self.actor_policy = actor_policy
        self.residual_logit_limit = float(residual_logit_limit)
        self.candidate_representation = candidate_representation
        self.parent_projection = nn.Sequential(
            nn.Linear(parent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.candidate_projection = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.global_projection = nn.Sequential(
            nn.Linear(global_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.operator_embedding = nn.Embedding(action_count, hidden_dim)
        self.network_encoder = (
            NetworkTransformerEncoder(
                graph_node_dim, graph_edge_dim, hidden_dim, heads, layers, dropout)
            if graph_node_dim is not None and graph_edge_dim is not None else None)
        self.candidate_path_encoder = (
            OrderedCandidatePathEncoder(
                hidden_dim, heads, dropout, use_attention=True)
            if self.network_encoder is not None
            and candidate_representation in (
                "path_v2", "path_v3", "path_v4") else None)
        encoder_layer = lambda: nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.parent_encoder = nn.TransformerEncoder(encoder_layer(), num_layers=layers)
        self.candidate_encoder = nn.TransformerEncoder(encoder_layer(), num_layers=layers)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.heads = _Heads(
            hidden_dim, action_count,
            critic_contexts=4 if self.network_encoder else 3,
            zero_actor=actor_policy == "base_residual")

    def forward(self, batch: Dict[str, torch.Tensor]):
        parent = self.parent_projection(_signed_log1p(batch["parent"]))
        candidate = self.candidate_projection(_signed_log1p(batch["candidates"]))
        global_hidden = self.global_projection(_signed_log1p(batch["global"]))
        network_hidden = None
        if self.network_encoder is not None:
            network_nodes, network_edges, network_hidden = self.network_encoder(batch)
            network_context = network_hidden.unsqueeze(1)
            parent = parent + network_context
            candidate_graph = (
                self.candidate_path_encoder(network_nodes, network_edges, batch)
                if self.candidate_path_encoder is not None else
                _candidate_graph_context(network_nodes, batch))
            candidate = candidate + network_context + candidate_graph
        operator = self.operator_embedding(batch["operator_id"]).unsqueeze(1)
        candidate = candidate + operator + global_hidden.unsqueeze(1)
        parent = self.parent_encoder(parent, src_key_padding_mask=batch["parent_padding_mask"])
        candidate = self.candidate_encoder(
            candidate, src_key_padding_mask=batch["candidate_padding_mask"])
        attended, _ = self.cross_attention(
            query=candidate, key=parent, value=parent,
            key_padding_mask=batch["parent_padding_mask"], need_weights=False)
        candidate = self.cross_norm(candidate + attended)
        logits = self.heads.actor_logits(candidate, batch["operator_id"])
        logits = _apply_actor_policy(
            logits, batch, self.actor_policy, self.residual_logit_limit)
        valid = batch["eligible_mask"] & ~batch["candidate_padding_mask"]
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        parent_pool = _masked_mean(parent, batch["parent_padding_mask"])
        candidate_pool = _masked_mean(candidate, batch["candidate_padding_mask"])
        contexts = [parent_pool, candidate_pool, global_hidden]
        if network_hidden is not None:
            contexts.append(network_hidden)
        value = self.heads.critic_head(torch.cat(contexts, dim=-1)).squeeze(-1)
        return logits, value


class MLPActorCritic(nn.Module):
    """Fair non-attention ablation using the same tokens, mask and PPO."""

    architecture = "mlp"

    def __init__(self, hidden_dim=128,
                 parent_dim=len(PARENT_FEATURES),
                 candidate_dim=len(CANDIDATE_FEATURES),
                 global_dim=len(GLOBAL_FEATURES), action_count=len(OPERATORS),
                 graph_node_dim=None, graph_edge_dim=None,
                 actor_policy="direct", residual_logit_limit=2.0,
                 candidate_representation="legacy", **_):
        super().__init__()
        self.actor_policy = actor_policy
        self.residual_logit_limit = float(residual_logit_limit)
        self.candidate_representation = candidate_representation
        self.parent_projection = nn.Sequential(
            nn.Linear(parent_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.network_encoder = (
            NetworkMeanEncoder(graph_node_dim, graph_edge_dim, hidden_dim)
            if graph_node_dim is not None and graph_edge_dim is not None else None)
        self.candidate_path_encoder = (
            OrderedCandidatePathEncoder(
                hidden_dim, heads=1, dropout=0.0, use_attention=False)
            if self.network_encoder is not None
            and self.candidate_representation in (
                "path_v2", "path_v3", "path_v4") else None)
        self.candidate_projection = nn.Sequential(
            nn.Linear(candidate_dim + global_dim + hidden_dim
                      + (hidden_dim if self.network_encoder else 0),
                      hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.global_projection = nn.Sequential(
            nn.Linear(global_dim, hidden_dim), nn.GELU())
        self.operator_embedding = nn.Embedding(action_count, hidden_dim)
        self.heads = _Heads(
            hidden_dim, action_count,
            critic_contexts=4 if self.network_encoder else 3,
            zero_actor=actor_policy == "base_residual")

    def forward(self, batch: Dict[str, torch.Tensor]):
        parent = self.parent_projection(_signed_log1p(batch["parent"]))
        parent_pool = _masked_mean(parent, batch["parent_padding_mask"])
        count = batch["candidates"].shape[1]
        global_raw = _signed_log1p(batch["global"])
        parts = [
            _signed_log1p(batch["candidates"]),
            global_raw.unsqueeze(1).expand(-1, count, -1),
            parent_pool.unsqueeze(1).expand(-1, count, -1),
        ]
        network_hidden = None
        if self.network_encoder is not None:
            network_nodes, network_edges, network_hidden = self.network_encoder(batch)
            candidate_graph = (
                self.candidate_path_encoder(network_nodes, network_edges, batch)
                if self.candidate_path_encoder is not None else
                _candidate_graph_context(network_nodes, batch))
            parts.append(network_hidden.unsqueeze(1).expand(-1, count, -1)
                         + candidate_graph)
        inputs = torch.cat(parts, dim=-1)
        candidate = self.candidate_projection(inputs)
        candidate = candidate + self.operator_embedding(batch["operator_id"]).unsqueeze(1)
        logits = self.heads.actor_logits(candidate, batch["operator_id"])
        logits = _apply_actor_policy(
            logits, batch, self.actor_policy, self.residual_logit_limit)
        valid = batch["eligible_mask"] & ~batch["candidate_padding_mask"]
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        candidate_pool = _masked_mean(candidate, batch["candidate_padding_mask"])
        global_hidden = self.global_projection(global_raw)
        contexts = [parent_pool, candidate_pool, global_hidden]
        if network_hidden is not None:
            contexts.append(network_hidden)
        value = self.heads.critic_head(torch.cat(contexts, dim=-1)).squeeze(-1)
        return logits, value


def build_model(architecture, config, model_spec=None):
    kwargs = dict(hidden_dim=config.hidden_dim, heads=config.attention_heads,
                  layers=config.transformer_layers, dropout=config.dropout,
                  actor_policy=config.actor_policy,
                  residual_logit_limit=config.residual_logit_limit,
                  candidate_representation=config.candidate_representation)
    if model_spec is not None:
        kwargs.update(model_spec)
    if architecture == "transformer":
        return TransformerActorCritic(**kwargs)
    if architecture == "mlp":
        return MLPActorCritic(**kwargs)
    raise ValueError(f"unknown architecture: {architecture}")
