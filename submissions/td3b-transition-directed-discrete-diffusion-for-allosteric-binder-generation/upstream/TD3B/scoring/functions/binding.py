import sys
import os, torch
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import esm
from transformers import AutoModelForMaskedLM


def _sanitize_token_ids(input_ids: torch.Tensor, vocab_size: int, unk_id: int) -> torch.Tensor:
    if vocab_size <= 0 or input_ids.numel() == 0:
        return input_ids
    if torch.any(input_ids >= vocab_size) or torch.any(input_ids < 0):
        # Replace out-of-range IDs with UNK to avoid embedding OOB.
        unk = torch.tensor(unk_id, device=input_ids.device, dtype=input_ids.dtype)
        input_ids = torch.where((input_ids >= vocab_size) | (input_ids < 0), unk, input_ids)
    return input_ids

class ImprovedBindingPredictor(nn.Module):
    def __init__(self,
                 esm_dim=1280,
                 smiles_dim=768,
                 hidden_dim=512,
                 n_heads=8,
                 n_layers=3,
                 dropout=0.1):
        super().__init__()

        # Define binding thresholds
        self.tight_threshold = 7.5    # Kd/Ki/IC50 ≤ ~30nM
        self.weak_threshold = 6.0     # Kd/Ki/IC50 > 1μM

        # Project to same dimension
        self.smiles_projection = nn.Linear(smiles_dim, hidden_dim)
        self.protein_projection = nn.Linear(esm_dim, hidden_dim)
        self.protein_norm = nn.LayerNorm(hidden_dim)
        self.smiles_norm = nn.LayerNorm(hidden_dim)

        # Cross attention blocks with layer norm
        self.cross_attention_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout),
                'norm1': nn.LayerNorm(hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim)
                ),
                'norm2': nn.LayerNorm(hidden_dim)
            }) for _ in range(n_layers)
        ])

        # Prediction heads
        self.shared_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Regression head
        self.regression_head = nn.Linear(hidden_dim, 1)

        # Classification head (3 classes: tight, medium, loose binding)
        self.classification_head = nn.Linear(hidden_dim, 3)

    def get_binding_class(self, affinity):
        """Convert affinity values to class indices
        0: tight binding (>= 7.5)
        1: medium binding (6.0-7.5)
        2: weak binding (< 6.0)
        """
        if isinstance(affinity, torch.Tensor):
            tight_mask = affinity >= self.tight_threshold
            weak_mask = affinity < self.weak_threshold
            medium_mask = ~(tight_mask | weak_mask)

            classes = torch.zeros_like(affinity, dtype=torch.long)
            classes[medium_mask] = 1
            classes[weak_mask] = 2
            return classes
        else:
            if affinity >= self.tight_threshold:
                return 0  # tight binding
            elif affinity < self.weak_threshold:
                return 2  # weak binding
            else:
                return 1  # medium binding

    def forward(self, protein_emb, smiles_emb):
        protein = self.protein_norm(self.protein_projection(protein_emb))
        smiles = self.smiles_norm(self.smiles_projection(smiles_emb))

        #protein = protein.transpose(0, 1)
        #smiles = smiles.transpose(0, 1)

        # Cross attention layers
        for layer in self.cross_attention_layers:
            # Protein attending to SMILES
            attended_protein = layer['attention'](
                protein, smiles, smiles
            )[0]
            protein = layer['norm1'](protein + attended_protein)
            protein = layer['norm2'](protein + layer['ffn'](protein))

            # SMILES attending to protein
            attended_smiles = layer['attention'](
                smiles, protein, protein
            )[0]
            smiles = layer['norm1'](smiles + attended_smiles)
            smiles = layer['norm2'](smiles + layer['ffn'](smiles))

        # Get sequence-level representations
        protein_pool = torch.mean(protein, dim=0)
        smiles_pool = torch.mean(smiles, dim=0)

        # Concatenate both representations
        combined = torch.cat([protein_pool, smiles_pool], dim=-1)

        # Shared features
        shared_features = self.shared_head(combined)

        regression_output = self.regression_head(shared_features)
        classification_logits = self.classification_head(shared_features)

        return regression_output, classification_logits

class BindingAffinity:
    def __init__(self, prot_seq, tokenizer, base_path, device=None, emb_model=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        # peptide embeddings
        if emb_model is not None:
            self.pep_model = emb_model.to(self.device).eval()
        else:
            self.pep_model = AutoModelForMaskedLM.from_pretrained('aaronfeller/PeptideCLM-23M-all').roformer.to(self.device).eval()

        self.pep_tokenizer = tokenizer
        self.unk_id = getattr(self.pep_tokenizer, "unk_token_id", None)
        if self.unk_id is None:
            self.unk_id = self.pep_tokenizer.vocab.get(self.pep_tokenizer.unk_token, 0)
        self.pep_vocab_size = None
        self.max_pep_len = None
        if hasattr(self.pep_model, "model") and hasattr(self.pep_model.model, "roformer"):
            self.pep_vocab_size = self.pep_model.model.roformer.embeddings.word_embeddings.num_embeddings
            self.max_pep_len = self.pep_model.model.roformer.config.max_position_embeddings
        elif hasattr(self.pep_model, "roformer"):
            self.pep_vocab_size = self.pep_model.roformer.embeddings.word_embeddings.num_embeddings
            self.max_pep_len = self.pep_model.roformer.config.max_position_embeddings
        elif hasattr(self.pep_model, "get_input_embeddings"):
            self.pep_vocab_size = self.pep_model.get_input_embeddings().num_embeddings
            self.max_pep_len = getattr(self.pep_model.config, "max_position_embeddings", None)

        self.model = ImprovedBindingPredictor().to(self.device)
        checkpoint = torch.load(f'{base_path}/scoring/functions/classifiers/binding-affinity.pt',
                                map_location=self.device,
                                weights_only=False)
        _bind_sd = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint)) \
            if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(_bind_sd)

        self.model.eval()

        self.esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()  # load ESM-2 model
        self.esm_model = self.esm_model.to(self.device).eval()
        self.prot_tokenizer = alphabet.get_batch_converter() # load esm tokenizer

        data = [("target", prot_seq)]
        # get tokenized protein
        _, _, prot_tokens = self.prot_tokenizer(data)
        prot_tokens = prot_tokens.to(self.device)
        with torch.no_grad():
            results = self.esm_model.forward(prot_tokens, repr_layers=[33])  # Example with ESM-2
            prot_emb = results["representations"][33]

        self.prot_emb = prot_emb[0].to(self.device)
        self.prot_emb = torch.mean(self.prot_emb, dim=0, keepdim=True)


    def forward(self, input_seqs):
        with torch.no_grad():
            scores = []
            for seq in input_seqs:
                pep_tokens = self.pep_tokenizer(
                    seq,
                    return_tensors='pt',
                    padding=True,
                    truncation=self.max_pep_len is not None,
                    max_length=self.max_pep_len,
                )

                pep_tokens = {k: v.to(self.device) for k, v in pep_tokens.items()}
                pep_tokens["input_ids"] = _sanitize_token_ids(
                    pep_tokens["input_ids"], int(self.pep_vocab_size or 0), int(self.unk_id)
                )

                with torch.no_grad():
                    # Check if using custom Roformer wrapper or standard model
                    if hasattr(self.pep_model, 'model'):
                        # Custom roformer.Roformer wrapper - get hidden states from inner model
                        emb = self.pep_model.model.roformer(
                            input_ids=pep_tokens['input_ids'],
                            attention_mask=pep_tokens.get('attention_mask'),
                            output_hidden_states=True
                        )
                        pep_emb = emb.last_hidden_state.squeeze(0)
                        pep_emb = torch.mean(pep_emb, dim=0, keepdim=True)
                    else:
                        # Standard AutoModelForMaskedLM
                        emb = self.pep_model(
                            input_ids=pep_tokens['input_ids'],
                            attention_mask=pep_tokens.get('attention_mask'),
                            output_hidden_states=True
                        )
                        pep_emb = emb.last_hidden_state.squeeze(0)
                        pep_emb = torch.mean(pep_emb, dim=0, keepdim=True)

                score, logits = self.model.forward(self.prot_emb, pep_emb)
                scores.append(score.item())
        return scores

    def __call__(self, input_seqs: list):
        return self.forward(input_seqs)


class MultiTargetBindingAffinity:
    """
    Binding affinity predictor that can handle multiple protein targets dynamically.

    Unlike BindingAffinity which pre-computes a single target's embedding,
    this class can switch between different protein targets on-the-fly.
    """

    def __init__(self, tokenizer, base_path, device=None, emb_model=None):
        """
        Initialize multi-target binding affinity predictor.

        Args:
            tokenizer: Peptide tokenizer
            base_path: Base path for model files
            device: Device for computation (default: auto-detect)
            emb_model: Optional pre-loaded embedding model
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        # Peptide embeddings
        if emb_model is not None:
            self.pep_model = emb_model.to(self.device).eval()
        else:
            self.pep_model = AutoModelForMaskedLM.from_pretrained('aaronfeller/PeptideCLM-23M-all').roformer.to(self.device).eval()

        self.pep_tokenizer = tokenizer
        self.unk_id = getattr(self.pep_tokenizer, "unk_token_id", None)
        if self.unk_id is None:
            self.unk_id = self.pep_tokenizer.vocab.get(self.pep_tokenizer.unk_token, 0)
        self.pep_vocab_size = None
        self.max_pep_len = None
        if hasattr(self.pep_model, "model") and hasattr(self.pep_model.model, "roformer"):
            self.pep_vocab_size = self.pep_model.model.roformer.embeddings.word_embeddings.num_embeddings
            self.max_pep_len = self.pep_model.model.roformer.config.max_position_embeddings
        elif hasattr(self.pep_model, "roformer"):
            self.pep_vocab_size = self.pep_model.roformer.embeddings.word_embeddings.num_embeddings
            self.max_pep_len = self.pep_model.roformer.config.max_position_embeddings
        elif hasattr(self.pep_model, "get_input_embeddings"):
            self.pep_vocab_size = self.pep_model.get_input_embeddings().num_embeddings
            self.max_pep_len = getattr(self.pep_model.config, "max_position_embeddings", None)

        # Binding affinity prediction model
        self.model = ImprovedBindingPredictor().to(self.device)
        checkpoint = torch.load(f'{base_path}/scoring/functions/classifiers/binding-affinity.pt',
                                map_location=self.device,
                                weights_only=False)
        _bind_sd = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint)) \
            if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(_bind_sd)
        self.model.eval()

        # Protein (ESM) model
        self.esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm_model = self.esm_model.to(self.device).eval()
        self.prot_tokenizer = alphabet.get_batch_converter()

        # Cache for protein embeddings (target_seq -> embedding)
        self.prot_emb_cache = {}

    def get_protein_embedding(self, prot_seq: str):
        """
        Get protein embedding, using cache if available.

        Args:
            prot_seq: Protein amino acid sequence

        Returns:
            Protein embedding tensor
        """
        # Check cache first
        if prot_seq in self.prot_emb_cache:
            return self.prot_emb_cache[prot_seq]

        # Compute embedding
        data = [("target", prot_seq)]
        _, _, prot_tokens = self.prot_tokenizer(data)
        prot_tokens = prot_tokens.to(self.device)

        with torch.no_grad():
            results = self.esm_model.forward(prot_tokens, repr_layers=[33])
            prot_emb = results["representations"][33]

        prot_emb = prot_emb[0].to(self.device)
        prot_emb = torch.mean(prot_emb, dim=0, keepdim=True)

        # Cache for future use
        self.prot_emb_cache[prot_seq] = prot_emb

        return prot_emb

    def forward(self, input_seqs, prot_seq: str):
        """
        Predict binding affinity for peptide-protein pairs.

        Args:
            input_seqs: List of peptide sequences
            prot_seq: Protein target sequence

        Returns:
            List of binding affinity scores
        """
        # Get protein embedding (cached if previously computed)
        prot_emb = self.get_protein_embedding(prot_seq)

        with torch.no_grad():
            scores = []
            for seq in input_seqs:
                pep_tokens = self.pep_tokenizer(
                    seq,
                    return_tensors='pt',
                    padding=True,
                    truncation=self.max_pep_len is not None,
                    max_length=self.max_pep_len,
                )
                pep_tokens = {k: v.to(self.device) for k, v in pep_tokens.items()}
                pep_tokens["input_ids"] = _sanitize_token_ids(
                    pep_tokens["input_ids"], int(self.pep_vocab_size or 0), int(self.unk_id)
                )

                with torch.no_grad():
                    # Check if using custom Roformer wrapper or standard model
                    if hasattr(self.pep_model, 'model'):
                        # Custom roformer.Roformer wrapper - get hidden states from inner model
                        emb = self.pep_model.model.roformer(
                            input_ids=pep_tokens['input_ids'],
                            attention_mask=pep_tokens.get('attention_mask'),
                            output_hidden_states=True
                        )
                        pep_emb = emb.last_hidden_state.squeeze(0)
                        pep_emb = torch.mean(pep_emb, dim=0, keepdim=True)
                    else:
                        # Standard AutoModelForMaskedLM
                        emb = self.pep_model(
                            input_ids=pep_tokens['input_ids'],
                            attention_mask=pep_tokens.get('attention_mask'),
                            output_hidden_states=True
                        )
                        pep_emb = emb.last_hidden_state.squeeze(0)
                        pep_emb = torch.mean(pep_emb, dim=0, keepdim=True)

                score, logits = self.model.forward(prot_emb, pep_emb)
                scores.append(score.item())

        return scores

    def forward_from_probs(
        self,
        token_probs: torch.Tensor,
        attention_mask: torch.Tensor,
        prot_seq: str,
    ) -> torch.Tensor:
        """
        Differentiable binding affinity from token probabilities.
        """
        if token_probs.dim() == 2:
            token_probs = token_probs.unsqueeze(0)
        token_probs = token_probs.to(self.device)
        attention_mask = attention_mask.to(self.device)

        roformer = None
        if hasattr(self.pep_model, "model") and hasattr(self.pep_model.model, "roformer"):
            roformer = self.pep_model.model.roformer
            emb_weight = roformer.embeddings.word_embeddings.weight
        elif hasattr(self.pep_model, "roformer"):
            roformer = self.pep_model.roformer
            emb_weight = roformer.embeddings.word_embeddings.weight
        else:
            emb_weight = self.pep_model.get_input_embeddings().weight

        if token_probs.size(-1) != emb_weight.size(0):
            raise ValueError(
                f"Token vocab mismatch: probs={token_probs.size(-1)} vs model={emb_weight.size(0)}"
            )

        inputs_embeds = token_probs @ emb_weight
        if roformer is not None:
            outputs = roformer(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state
        else:
            outputs = self.pep_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1]

        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
        pep_emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        prot_emb = self.get_protein_embedding(prot_seq).to(self.device)
        prot_emb = prot_emb.expand(pep_emb.size(0), -1).unsqueeze(0)
        pep_emb = pep_emb.unsqueeze(0)

        score, _ = self.model.forward(prot_emb, pep_emb)
        return score.squeeze(-1)

    def __call__(self, input_seqs: list, prot_seq: str):
        """
        Predict binding affinity for peptide-protein pairs.

        Args:
            input_seqs: List of peptide sequences
            prot_seq: Protein target sequence

        Returns:
            List of binding affinity scores
        """
        return self.forward(input_seqs, prot_seq)

    def clear_cache(self):
        """Clear the protein embedding cache to free memory."""
        self.prot_emb_cache = {}


class TargetSpecificBindingAffinity:
    """
    Wrapper that binds a specific protein target to MultiTargetBindingAffinity.

    This allows using MultiTargetBindingAffinity with the standard BindingAffinity interface
    where only peptide sequences need to be provided.
    """

    def __init__(self, multi_target_predictor: MultiTargetBindingAffinity, prot_seq: str):
        """
        Create a target-specific binding affinity predictor.

        Args:
            multi_target_predictor: The underlying multi-target predictor
            prot_seq: The protein target sequence to use
        """
        self.predictor = multi_target_predictor
        self.prot_seq = prot_seq

    def forward(self, input_seqs):
        """
        Predict binding affinity for peptides against the bound target.

        Args:
            input_seqs: List of peptide sequences

        Returns:
            List of binding affinity scores
        """
        return self.predictor.forward(input_seqs, self.prot_seq)

    def __call__(self, input_seqs: list):
        """
        Predict binding affinity for peptides against the bound target.

        Args:
            input_seqs: List of peptide sequences

        Returns:
            List of binding affinity scores
        """
        return self.forward(input_seqs)
