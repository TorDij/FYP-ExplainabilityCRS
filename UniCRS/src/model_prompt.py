# model_prompt.py
import math
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv


class KGPrompt(nn.Module):
    def __init__(
        self, hidden_size, token_hidden_size, n_head, n_layer, n_block,
        n_entity, num_relations, num_bases, edge_index, edge_type,
        n_prefix_rec=None, n_prefix_conv=None
    ):
        super(KGPrompt, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_prefix_rec = n_prefix_rec
        self.n_prefix_conv = n_prefix_conv
        self.num_relations = num_relations

        entity_hidden_size = hidden_size // 2
        self.kg_encoder = RGCNConv(entity_hidden_size, entity_hidden_size, num_relations=num_relations,
                                   num_bases=num_bases)
        self.node_embeds = nn.Parameter(torch.empty(n_entity, entity_hidden_size))
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)
        self.edge_index = nn.Parameter(edge_index, requires_grad=False)
        self.edge_type = nn.Parameter(edge_type, requires_grad=False)
        self.entity_proj1 = nn.Sequential(
            nn.Linear(entity_hidden_size, entity_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(entity_hidden_size // 2, entity_hidden_size),
        )
        self.entity_proj2 = nn.Linear(entity_hidden_size, hidden_size)

        self.token_proj1 = nn.Sequential(
            nn.Linear(token_hidden_size, token_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(token_hidden_size // 2, token_hidden_size),
        )
        self.token_proj2 = nn.Linear(token_hidden_size, hidden_size)

        self.cross_attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.prompt_proj1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.prompt_proj2 = nn.Linear(hidden_size, n_layer * n_block * hidden_size)

        if self.n_prefix_rec is not None:
            self.rec_prefix_embeds = nn.Parameter(torch.empty(n_prefix_rec, hidden_size))
            nn.init.normal_(self.rec_prefix_embeds)
            self.rec_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )
            # --------------- ADDED FOR ENHANCED REC PROMPT ----------------------
            # New projection layer for `relationship_embeddings`
            self.relationship_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )
            # --------------------------------------------------------------------

        if self.n_prefix_conv is not None:
            self.conv_prefix_embeds = nn.Parameter(torch.empty(n_prefix_conv, hidden_size))
            nn.init.normal_(self.conv_prefix_embeds)
            self.conv_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )

    def set_and_fix_node_embed(self, node_embeds: torch.Tensor):
        self.node_embeds.data = node_embeds
        self.node_embeds.requires_grad_(False)

    @staticmethod
    def validate_rgcn_inputs(node_embeds, edge_index, edge_type, num_relations):
        """
        Validates inputs for RGCN to catch common errors
        """
        num_nodes = node_embeds.size(0)
        
        # Check edge_index bounds
        if edge_index.max() >= num_nodes:
            raise ValueError(f"Edge index contains node references ({edge_index.max()}) >= number of nodes ({num_nodes})")
        if edge_index.min() < 0:
            raise ValueError(f"Edge index contains negative values ({edge_index.min()})")
            
        # Check edge_type bounds
        if edge_type.max() >= num_relations:
            raise ValueError(f"Edge type contains relation ids ({edge_type.max()}) >= number of relations ({num_relations})")
        if edge_type.min() < 0:
            raise ValueError(f"Edge type contains negative values ({edge_type.min()})")
            
        # Check shapes
        if edge_index.size(1) != edge_type.size(0):
            raise ValueError(f"Number of edges in edge_index ({edge_index.size(1)}) != number of edge types ({edge_type.size(0)})")
        
        # Print tensor device information
        # print(f"node_embeds device: {node_embeds.device}")
        # print(f"edge_index device: {edge_index.device}")
        # print(f"edge_type device: {edge_type.device}")
        
        # Print shape information
        # print(f"node_embeds shape: {node_embeds.shape}")
        # print(f"edge_index shape: {edge_index.shape}")
        # print(f"edge_type shape: {edge_type.shape}")
        
        return True
    
    
    def get_entity_embeds(self):
        node_embeds = self.node_embeds
        
        # Add validation checks
        self.validate_rgcn_inputs(
            node_embeds=node_embeds,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            num_relations=self.num_relations
        )
        
        # Ensure all tensors are on the same device
        device = node_embeds.device
        edge_index = self.edge_index.to(device)
        edge_type = self.edge_type.to(device)
        
        # Original processing
        entity_embeds = self.kg_encoder(node_embeds, edge_index, edge_type) + node_embeds
        entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        entity_embeds = self.entity_proj2(entity_embeds)
        return entity_embeds


    # def get_entity_embeds(self):
    #     node_embeds = self.node_embeds
    #     entity_embeds = self.kg_encoder(node_embeds, self.edge_index, self.edge_type) + node_embeds
    #     entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
    #     entity_embeds = self.entity_proj2(entity_embeds)
    #     return entity_embeds

    def forward(self, entity_ids=None, token_embeds=None,relationship_embeddings=None, output_entity=False, use_rec_prefix=False,
                use_conv_prefix=False):
        batch_size, entity_embeds, entity_len, token_len = None, None, None, None
        device = next(self.parameters()).device 
        
        if entity_ids is not None:
            batch_size, entity_len = entity_ids.shape[:2]
            entity_embeds = self.get_entity_embeds()
            entity_embeds = entity_embeds[entity_ids]  # (batch_size, entity_len, hidden_size)
        if token_embeds is not None:
            batch_size, token_len = token_embeds.shape[:2]
            token_embeds = self.token_proj1(token_embeds) + token_embeds  # (batch_size, token_len, hidden_size)
            token_embeds = self.token_proj2(token_embeds)

        if entity_embeds is not None and token_embeds is not None:
            attn_weights = self.cross_attn(token_embeds) @ entity_embeds.permute(0, 2,
                                                                                 1)  # (batch_size, token_len, entity_len)
            attn_weights /= self.hidden_size

            if output_entity:
                token_weights = F.softmax(attn_weights, dim=1).permute(0, 2, 1)
                prompt_embeds = token_weights @ token_embeds + entity_embeds
                prompt_len = entity_len
            else:
                entity_weights = F.softmax(attn_weights, dim=2)
                prompt_embeds = entity_weights @ entity_embeds + token_embeds
                prompt_len = token_len
        elif entity_embeds is not None:
            prompt_embeds = entity_embeds
            prompt_len = entity_len
        else:
            prompt_embeds = token_embeds
            prompt_len = token_len

        if self.n_prefix_rec is not None and use_rec_prefix:
            prefix_embeds = self.rec_prefix_proj(self.rec_prefix_embeds) + self.rec_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_rec

        # --------------- ADDED FOR ENHANCED REC PROMPT ----------------------
        # **STEP 2: Add Relationship Embeddings to the Recommendation Prompt**
        if relationship_embeddings is not None:
            relationship_embeddings = relationship_embeddings.to(device)  # Move to the correct device

            # Ensure correct shape: (batch_size, seq_len, hidden_size)
            if relationship_embeddings.dim() == 2:  # If (batch_size, hidden_size), add seq_len dimension
                relationship_embeddings = relationship_embeddings.unsqueeze(1)  # (batch_size, 1, hidden_size)
            elif relationship_embeddings.dim() == 1:  # If (hidden_size,), add batch and seq_len dimensions
                relationship_embeddings = relationship_embeddings.unsqueeze(0).unsqueeze(1)  # (1, 1, hidden_size)

            # **Ensure hidden size is correct before projection**
            if relationship_embeddings.shape[-1] != self.hidden_size:
                if relationship_embeddings.shape[-1] > self.hidden_size:
                    print("[WARNING] Truncating relationship_embeddings to match hidden_size.")
                    relationship_embeddings = relationship_embeddings[..., :self.hidden_size]  # Trim excess dimensions
                else:
                    print("[WARNING] Padding relationship_embeddings to match hidden_size.")
                    relationship_embeddings = torch.nn.functional.pad(
                        relationship_embeddings,
                        (0, self.hidden_size - relationship_embeddings.shape[-1])
                    )

            # Apply linear projection
            relationship_embeds = self.relationship_proj(relationship_embeddings)
            # Debugging the shape before expansion
            # print(f"[DEBUG] Relationship Embeds Shape Before Expansion: {relationship_embeds.shape}")
    
            # relationship_embeds = relationship_embeds.expand(prompt_embeds.shape[0], -1, -1)  # Ensure batch_size consistency
            relationship_embeds = relationship_embeds.expand_as(prompt_embeds)

            # Ensure relationship_embeds matches prompt_embeds in seq_len
            if relationship_embeds.shape[1] != prompt_embeds.shape[1]:
                min_seq_len = min(relationship_embeds.shape[1], prompt_embeds.shape[1])
                relationship_embeds = relationship_embeds[:, :min_seq_len, :]
                prompt_embeds = prompt_embeds[:, :min_seq_len, :]

            # print(f"[DEBUG] Relationship Embeddings Shape After Expansion: {relationship_embeds.shape}")
            # print(f"[DEBUG] Prompt Embeddings Shape Before Concatenation: {prompt_embeds.shape}")

            # Ensure they have the same dimensions before concatenation
            assert relationship_embeds.shape[1:] == prompt_embeds.shape[1:], \
                f"Shape mismatch: {relationship_embeds.shape} vs {prompt_embeds.shape}"

            prompt_embeds = prompt_embeds + relationship_embeds  # ✅ Element-wise addition, keeps hidden_size = 768



        # --------------------------------------------------------------------
            
        if self.n_prefix_conv is not None and use_conv_prefix:
            prefix_embeds = self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_conv


        prompt_embeds = self.prompt_proj1(prompt_embeds) + prompt_embeds
        prompt_embeds = self.prompt_proj2(prompt_embeds)

        # print(f"[DEBUG] prompt_embeds shape BEFORE reshaping: {prompt_embeds.shape}")
        # print(f"[DEBUG] Expected elements: {batch_size * prompt_len * self.n_layer * self.n_block * self.n_head * self.head_dim}")
        # print(f"[DEBUG] Actual elements: {prompt_embeds.numel()}")
        
        expected_size = batch_size * prompt_len * self.n_layer * self.n_block * self.n_head * self.head_dim
        if prompt_embeds.numel() != expected_size:
            raise ValueError(f"Mismatch in elements: expected {expected_size}, but got {prompt_embeds.numel()}")

        prompt_embeds = prompt_embeds.reshape(
            batch_size, prompt_len, self.n_layer, self.n_block, self.n_head, self.head_dim
        ).permute(2, 3, 0, 4, 1, 5) # (n_layer, n_block, batch_size, n_head, prompt_len, head_dim)


        return prompt_embeds

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        state_dict = {k: v for k, v in self.state_dict().items() if 'edge' not in k}
        save_path = os.path.join(save_dir, 'model.pt')
        torch.save(state_dict, save_path)

    def load(self, load_dir):
        load_path = os.path.join(load_dir, 'model.pt')
        missing_keys, unexpected_keys = self.load_state_dict(
            torch.load(load_path, map_location=torch.device('cpu')), strict=False
        )
        print(missing_keys, unexpected_keys)
