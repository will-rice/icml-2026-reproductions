"""
TD3B Data Utilities
Handles loading and preprocessing of TD3B_data.csv for both oracle training and finetuning.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
import sys

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - rdkit may be optional in some setups
    Chem = None

sys.path.append('..')

AA_SET = set("ACDEFGHIKLMNPQRSTVWY")


def is_amino_acid_sequence(seq: str) -> bool:
    if not isinstance(seq, str) or not seq:
        return False
    seq = seq.strip().upper()
    return all(ch in AA_SET for ch in seq)


def aa_sequence_to_smiles(seq: str) -> Optional[str]:
    if Chem is None or not is_amino_acid_sequence(seq):
        return None
    try:
        mol = Chem.MolFromSequence(seq)
    except Exception:
        return None
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def peptide_seq_to_smiles(seq: str) -> str:
    smiles = aa_sequence_to_smiles(seq)
    return smiles if smiles is not None else seq


def smiles_token_length(smiles: str, tokenizer) -> int:
    if tokenizer is None:
        return len(smiles)
    tokens = tokenizer(smiles, return_tensors="pt")["input_ids"][0]
    return int(tokens.numel())


class TD3BDataset(Dataset):
    """
    Dataset for TD3B that loads peptide-protein pairs with directional labels.

    Supports both:
    1. Oracle training: uses all pairs for training f_φ
    2. Finetuning: provides target proteins for conditioning during RL
    """

    def __init__(
        self,
        data_path: str,
        mode: str = 'oracle',  # 'oracle' or 'finetune'
        peptide_tokenizer=None,
        protein_tokenizer=None,
        max_peptide_length: int = 200,
        max_protein_length: int = 1000,
        target_protein_id: Optional[str] = None,  # For finetuning mode
        convert_peptide_to_smiles: bool = True,
    ):
        """
        Args:
            data_path: Path to TD3B_data.csv
            mode: 'oracle' for training f_φ, 'finetune' for RL conditioning
            peptide_tokenizer: Tokenizer for peptide sequences
            protein_tokenizer: Tokenizer for protein sequences (ESM-2)
            max_peptide_length: Maximum peptide sequence length
            max_protein_length: Maximum protein sequence length
            target_protein_id: UniProt ID for target protein (finetuning mode)
        """
        self.mode = mode
        self.data_path = data_path
        self.peptide_tokenizer = peptide_tokenizer
        self.protein_tokenizer = protein_tokenizer
        self.max_peptide_length = max_peptide_length
        self.max_protein_length = max_protein_length
        self.convert_peptide_to_smiles = convert_peptide_to_smiles

        # Load data
        self.data = pd.read_csv(data_path)
        print(f"Loaded {len(self.data)} peptide-protein pairs from {data_path}")

        # Filter by target protein if in finetune mode
        if mode == 'finetune' and target_protein_id is not None:
            self.data = self.data[self.data['Target_UniProt_ID'] == target_protein_id]
            print(f"Filtered to {len(self.data)} pairs for target {target_protein_id}")

        # Process labels
        self.label_map = {
            'agonist': 1.0,
            'antagonist': -1.0,
            'neutral': 0.0,
        }

        # Convert action descriptions to numerical labels
        self.data['numeric_label'] = self.data['label'].map(self.label_map)

        # Assign confidence based on action description
        self.data['confidence'] = self.data['Action'].apply(self._action_to_confidence)

    def _action_to_confidence(self, action: str) -> float:
        """
        Convert action description to confidence score.

        Full agonist/antagonist: 1.0
        Partial/Weak: 0.7
        Others: 0.5
        """
        action_lower = action.lower()

        if 'full' in action_lower:
            return 1.0
        elif 'partial' in action_lower or 'weak' in action_lower:
            return 0.7
        elif 'slows' in action_lower or 'modulator' in action_lower:
            return 0.5
        else:
            return 0.8  # Default for unspecified agonist/antagonist

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Get sequences
        peptide_seq = row['Ligand_Sequence']
        protein_seq = row['Target_Sequence']
        peptide_smiles = self._peptide_to_smiles(peptide_seq)
        peptide_smiles_length = smiles_token_length(peptide_smiles, self.peptide_tokenizer)

        # Tokenize (placeholder - actual tokenization depends on mode)
        if self.peptide_tokenizer is not None:
            peptide_tokens = self._tokenize_peptide(peptide_smiles)
        else:
            peptide_tokens = torch.zeros(self.max_peptide_length, dtype=torch.long)

        if self.protein_tokenizer is not None:
            protein_tokens = self._tokenize_protein(protein_seq)
        else:
            protein_tokens = self._tokenize_protein_placeholder(protein_seq)

        # Get label and confidence
        label = torch.tensor(row['numeric_label'], dtype=torch.float32)
        confidence = torch.tensor(row['confidence'], dtype=torch.float32)

        return {
            'peptide_seq': peptide_seq,
            'peptide_smiles': peptide_smiles,
            'peptide_smiles_length': peptide_smiles_length,
            'protein_seq': protein_seq,
            'peptide_tokens': peptide_tokens,
            'protein_tokens': protein_tokens,
            'label': label,
            'confidence': confidence,
            'target_id': row['Target_UniProt_ID'],
            'ligand_id': row['Ligand_UniProt_ID'],
            'action': row['Action']
        }

    def _peptide_to_smiles(self, peptide_seq: str) -> str:
        if not self.convert_peptide_to_smiles:
            return peptide_seq
        return peptide_seq_to_smiles(peptide_seq)

    def _tokenize_peptide(self, peptide_seq: str) -> torch.Tensor:
        """Tokenize peptide sequence using provided tokenizer."""
        tokens = self.peptide_tokenizer(
            peptide_seq,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_peptide_length,
            truncation=True
        )['input_ids'].squeeze(0)
        return tokens

    def _tokenize_protein_placeholder(self, protein_seq: str) -> torch.Tensor:
        """
        Placeholder protein tokenizer (character-level).

        NOTE: Replace with ESM-2 tokenizer in production:
            from esm import pretrained
            _, alphabet = pretrained.esm2_t33_650M_UR50D()
            batch_converter = alphabet.get_batch_converter()
            _, _, tokens = batch_converter([("protein", protein_seq)])
        """
        # Amino acid to index mapping
        aa_to_idx = {aa: i+1 for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        aa_to_idx['<PAD>'] = 0
        aa_to_idx['<UNK>'] = 21

        # Convert to indices
        indices = [aa_to_idx.get(aa, aa_to_idx['<UNK>']) for aa in protein_seq]

        # Pad or truncate
        if len(indices) > self.max_protein_length:
            indices = indices[:self.max_protein_length]
        else:
            indices += [0] * (self.max_protein_length - len(indices))

        return torch.tensor(indices, dtype=torch.long)

    def _tokenize_protein(self, protein_seq: str) -> torch.Tensor:
        """Tokenize protein using ESM-2 tokenizer if available."""
        if self.protein_tokenizer is None:
            return self._tokenize_protein_placeholder(protein_seq)

        # Use ESM-2 tokenizer
        # TODO: Implement when ESM-2 is integrated
        return self._tokenize_protein_placeholder(protein_seq)

    def get_target_proteins(self) -> Dict[str, str]:
        """
        Get dictionary of unique target proteins.

        Returns:
            dict: {UniProt_ID: Sequence}
        """
        unique_targets = self.data.drop_duplicates(subset=['Target_UniProt_ID'])
        return dict(zip(unique_targets['Target_UniProt_ID'], unique_targets['Target_Sequence']))

    def get_ligands_for_target(self, target_id: str) -> List[Dict]:
        """
        Get all ligands (peptides) for a specific target protein.

        Args:
            target_id: Target protein UniProt ID

        Returns:
            List of dicts with ligand info
        """
        target_data = self.data[self.data['Target_UniProt_ID'] == target_id]

        ligands = []
        for _, row in target_data.iterrows():
            ligands.append({
                'sequence': row['Ligand_Sequence'],
                'uniprot_id': row['Ligand_UniProt_ID'],
                'label': row['numeric_label'],
                'confidence': row['confidence'],
                'action': row['Action']
            })

        return ligands


def load_td3b_data(
    data_path: str,
    mode: str = 'oracle',
    target_protein_id: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load and summarize TD3B data.

    Args:
        data_path: Path to TD3B_data.csv
        mode: 'oracle' or 'finetune'
        target_protein_id: Filter by target protein (finetuning mode)

    Returns:
        data: Filtered DataFrame
        stats: Dictionary of statistics
    """
    data = pd.read_csv(data_path)

    # Filter if needed
    if mode == 'finetune' and target_protein_id is not None:
        data = data[data['Target_UniProt_ID'] == target_protein_id]

    # Compute statistics
    stats = {
        'total_pairs': len(data),
        'unique_targets': data['Target_UniProt_ID'].nunique(),
        'unique_ligands': data['Ligand_UniProt_ID'].nunique(),
        'agonist_count': (data['label'] == 'agonist').sum(),
        'antagonist_count': (data['label'] == 'antagonist').sum(),
        'action_distribution': data['Action'].value_counts().to_dict()
    }

    return data, stats


def create_target_dataset_for_finetuning(
    data_path: str,
    target_protein_id: str,
    desired_direction: str = 'agonist'
) -> Dict:
    """
    Create a dataset for TD3B finetuning focused on a specific target.

    Args:
        data_path: Path to TD3B_data.csv
        target_protein_id: Target protein UniProt ID
        desired_direction: 'agonist' or 'antagonist'

    Returns:
        dict with target protein info and example ligands
    """
    data = pd.read_csv(data_path)

    # Get target protein info
    target_data = data[data['Target_UniProt_ID'] == target_protein_id]

    if len(target_data) == 0:
        raise ValueError(f"No data found for target {target_protein_id}")

    # Get protein sequence (should be same for all rows)
    protein_seq = target_data.iloc[0]['Target_Sequence']

    # Get ligands with desired direction
    direction_map = {'agonist': 'agonist', 'antagonist': 'antagonist'}
    direction_ligands = target_data[target_data['label'] == direction_map[desired_direction]]

    # Also get opposite direction for contrastive learning
    opposite_direction = 'antagonist' if desired_direction == 'agonist' else 'agonist'
    opposite_ligands = target_data[target_data['label'] == opposite_direction]

    return {
        'target_protein_id': target_protein_id,
        'target_protein_seq': protein_seq,
        'desired_direction': desired_direction,
        'n_desired_examples': len(direction_ligands),
        'n_opposite_examples': len(opposite_ligands),
        'desired_ligands': direction_ligands[['Ligand_Sequence', 'Action', 'Ligand_UniProt_ID']].to_dict('records'),
        'opposite_ligands': opposite_ligands[['Ligand_Sequence', 'Action', 'Ligand_UniProt_ID']].to_dict('records')
    }


if __name__ == "__main__":
    # Example usage
    data_path = "../TD3B_data.csv"

    print("=" * 80)
    print("TD3B Data Loading Example")
    print("=" * 80)

    # Load and summarize data
    data, stats = load_td3b_data(data_path, mode='oracle')

    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Create dataset for oracle training
    print("\n" + "=" * 80)
    print("Oracle Training Dataset")
    print("=" * 80)

    dataset = TD3BDataset(data_path, mode='oracle')
    print(f"Dataset size: {len(dataset)}")

    # Sample first item
    sample = dataset[0]
    print(f"\nSample item:")
    print(f"  Target: {sample['target_id']}")
    print(f"  Ligand: {sample['ligand_id']}")
    print(f"  Label: {sample['label'].item()}")
    print(f"  Confidence: {sample['confidence'].item()}")
    print(f"  Action: {sample['action']}")

    # Create finetuning dataset for a specific target
    print("\n" + "=" * 80)
    print("Finetuning Dataset Example")
    print("=" * 80)

    # Get first target
    targets = dataset.get_target_proteins()
    first_target_id = list(targets.keys())[0]

    finetune_info = create_target_dataset_for_finetuning(
        data_path,
        first_target_id,
        desired_direction='agonist'
    )

    print(f"\nTarget: {finetune_info['target_protein_id']}")
    print(f"Desired direction: {finetune_info['desired_direction']}")
    print(f"Number of agonist examples: {finetune_info['n_desired_examples']}")
    print(f"Number of antagonist examples: {finetune_info['n_opposite_examples']}")
