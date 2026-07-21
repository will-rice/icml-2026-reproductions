"""
Configuration sampler for sequencing pipeline.

Generates randomized configurations for SequencingEngine based on probability distributions.
"""
import random
from typing import Dict, Any, Optional, List, Tuple


class ConfigSampler:
    """Samples sequencing configurations based on probability distributions."""

    # Default probability distributions
    DEFAULT_DISTRIBUTIONS = {
        "OperationMerger": {
            "scope": {"global": 0.75, "per_region": 0.25},
            "strategies": {
                '["standard"]': 0.50,
                '["paste_detection", "standard"]': 0.50
            },
            "merge_params": {
                "row_first": {True: 0.60, False: 0.40},
                "merge_inputs": {True: 0.975, False: 0.025},
                "merge_inputs_for_data_tables_only": {True: 0.50, False: 0.50},
                "force_merge_pasted_ranges": {True: 0.75, False: 0.25},
                "sort_input_by_type": {True: 0.80, False: 0.20},
                "smart_merge_inputs": {True: 0.75, False: 0.25},
                "smart_merge_inputs_threshold": {16: 0.15, 32: 0.35, 64: 0.35, 128: 0.15},
            },
            "paste_detection_params": {
                "force_paste_type": {None: 0.70, "paste_format": 0.20, "paste_full": 0.10},
                "paste_execution": {"block": 0.50, "interleaved": 0.50},
                "paste_full_ordering": {"grouped": 0.50, "alternating": 0.50}
            }
        },
        "RegionOrchestrator": {
            "mode": {"dependency": 0.75, "parallel": 0.20, "sequential": 0.05},
            "sequential_ordering": {"row_first": 0.60, "col_first": 0.35, "id_order": 0.05}
        },
        "OperationSequencer": {
            "scope": {"per_region": 0.40, "global": 0.60},
            "ordering_strategy": {"custom": 1.00},
            "custom_key": {
                "(min_row, min_col, op_priority, max_row, max_col)": 0.60,
                "(min_col, min_row, op_priority, max_col, max_row)": 0.40
            },
            "tie_mode": {"deterministic_sub_order": 0.25, "sample_random": 0.75},
            "tie_seed": {42: 1.00},
            "respect_closing_operations": {True: 0.75, False: 0.25}
        },
        "ConstraintEnforcer": {
            "enabled": {True: 1.00}
        },
        "DateNoiseFilter": {
            "enabled": {True: 0.80, False: 0.20}
        },
        "BorderConsolidator": {
            "enabled": {True: 1.00}
        },
        "DefaultValueFilter": {
            "enabled": {True: 1.0}
        },
        "FormattingConsolidator": {
            "enabled": {True: 1.0}
        },
        "InputSplitter": {
            "enabled": {True: 1.0},
            "split_threshold": {1: 0.40, 2: 0.50, 4: 0.10},
            "sparse_density": {0.5: 1.0}
        },
        "AutoFillDetector": {
            "enabled": {True: 1.0},
            "detect_formulas": {True: 1.0},
            "detect_values": {True: 0.80, False: 0.20},
            "min_fill": {2: 1.0}
        }
    }

    def __init__(
        self,
        distributions: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the config sampler.

        Args:
            distributions: Custom probability distributions. If None, uses DEFAULT_DISTRIBUTIONS.
                          Partial overrides are merged with defaults.
            seed: Random seed for reproducibility. If None, uses system randomness.
        """
        self.distributions = self._merge_distributions(distributions)
        self.rng = random.Random(seed)

    def _merge_distributions(self, custom: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge custom distributions with defaults."""
        if custom is None:
            return self.DEFAULT_DISTRIBUTIONS.copy()

        merged = {}
        for transformer, default_params in self.DEFAULT_DISTRIBUTIONS.items():
            if transformer in custom:
                merged[transformer] = self._deep_merge(
                    default_params.copy(),
                    custom[transformer]
                )
            else:
                merged[transformer] = default_params.copy()

        for transformer in custom:
            if transformer not in merged:
                merged[transformer] = custom[transformer].copy()

        return merged

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Recursively merge override into base."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _weighted_choice(self, options: Dict[Any, float]) -> Any:
        """Select an option based on probability weights."""
        choices = list(options.keys())
        weights = list(options.values())

        total = sum(weights)
        normalized_weights = [w / total for w in weights]

        return self.rng.choices(choices, weights=normalized_weights, k=1)[0]

    def _parse_strategies(self, strategies_str: str) -> List[str]:
        """Parse strategies string representation to list."""
        cleaned = strategies_str.strip('[]').replace('"', '').replace("'", "")
        return [s.strip() for s in cleaned.split(',')]

    def sample(self) -> Dict[str, Any]:
        """
        Sample a complete sequencing configuration.

        Returns:
            A configuration dict suitable for SequencingEngine.
        """
        config = {"pipeline": []}

        # Sample OperationMerger
        if "OperationMerger" in self.distributions:
            merger_dist = self.distributions["OperationMerger"]
            merger_config = {}

            if "scope" in merger_dist:
                merger_config["scope"] = self._weighted_choice(merger_dist["scope"])

            if "strategies" in merger_dist:
                strategies_str = self._weighted_choice(merger_dist["strategies"])
                merger_config["strategies"] = self._parse_strategies(strategies_str)

            if "merge_params" in merger_dist:
                merge_params = {}
                for param, options in merger_dist["merge_params"].items():
                    merge_params[param] = self._weighted_choice(options)
                merger_config["merge_params"] = merge_params

            if "paste_detection_params" in merger_dist:
                paste_params = {}
                for param, options in merger_dist["paste_detection_params"].items():
                    paste_params[param] = self._weighted_choice(options)
                merger_config["paste_detection_params"] = paste_params

            config["pipeline"].append({
                "type": "OperationMerger",
                "config": merger_config
            })

        # Sample RegionOrchestrator
        if "RegionOrchestrator" in self.distributions:
            orchestrator_dist = self.distributions["RegionOrchestrator"]
            orchestrator_config = {}

            if "mode" in orchestrator_dist:
                orchestrator_config["mode"] = self._weighted_choice(orchestrator_dist["mode"])

            if "sequential_ordering" in orchestrator_dist:
                orchestrator_config["sequential_ordering"] = self._weighted_choice(
                    orchestrator_dist["sequential_ordering"]
                )

            config["pipeline"].append({
                "type": "RegionOrchestrator",
                "config": orchestrator_config
            })

        # Sample OperationSequencer
        if "OperationSequencer" in self.distributions:
            sequencer_dist = self.distributions["OperationSequencer"]
            sequencer_config = {}

            if "scope" in sequencer_dist:
                sequencer_config["scope"] = self._weighted_choice(sequencer_dist["scope"])

            if "ordering_strategy" in sequencer_dist:
                sequencer_config["ordering_strategy"] = self._weighted_choice(
                    sequencer_dist["ordering_strategy"]
                )

            if "custom_key" in sequencer_dist:
                sequencer_config["custom_key"] = self._weighted_choice(sequencer_dist["custom_key"])

            if "tie_mode" in sequencer_dist:
                sequencer_config["tie_mode"] = self._weighted_choice(sequencer_dist["tie_mode"])

            if "tie_seed" in sequencer_dist:
                sequencer_config["tie_seed"] = self._weighted_choice(sequencer_dist["tie_seed"])

            if "respect_closing_operations" in sequencer_dist:
                sequencer_config["respect_closing_operations"] = self._weighted_choice(
                    sequencer_dist["respect_closing_operations"]
                )

            config["pipeline"].append({
                "type": "OperationSequencer",
                "config": sequencer_config
            })

        # Sample ConstraintEnforcer
        if "ConstraintEnforcer" in self.distributions:
            enforcer_dist = self.distributions["ConstraintEnforcer"]
            enforcer_config = {}

            if "enabled" in enforcer_dist:
                enforcer_config["enabled"] = self._weighted_choice(enforcer_dist["enabled"])

            config["pipeline"].append({
                "type": "ConstraintEnforcer",
                "config": enforcer_config
            })

        # Sample DateNoiseFilter (inserted early to clean data before processing)
        if "DateNoiseFilter" in self.distributions:
            date_dist = self.distributions["DateNoiseFilter"]
            date_config = {}

            for param, options in date_dist.items():
                date_config[param] = self._weighted_choice(options)

            if date_config.get("enabled", True):
                config["pipeline"].insert(0, {
                    "type": "DateNoiseFilter",
                    "config": date_config
                })

        # Sample BorderConsolidator (inserted after merger, before sequencer)
        if "BorderConsolidator" in self.distributions:
            border_dist = self.distributions["BorderConsolidator"]
            border_config = {}

            for param, options in border_dist.items():
                border_config[param] = self._weighted_choice(options)

            merger_idx = next(
                (i for i, t in enumerate(config["pipeline"]) if t["type"] == "OperationMerger"),
                0
            )
            config["pipeline"].insert(merger_idx + 1, {
                "type": "BorderConsolidator",
                "config": border_config
            })

        # Sample DefaultValueFilter (inserted after merger, before sequencer)
        if "DefaultValueFilter" in self.distributions:
            default_dist = self.distributions["DefaultValueFilter"]
            default_config = {}

            for param, options in default_dist.items():
                default_config[param] = self._weighted_choice(options)

            if default_config.get("enabled", True):
                border_idx = next(
                    (i for i, t in enumerate(config["pipeline"]) if t["type"] == "BorderConsolidator"),
                    None
                )
                if border_idx is not None:
                    insert_idx = border_idx + 1
                else:
                    merger_idx = next(
                        (i for i, t in enumerate(config["pipeline"]) if t["type"] == "OperationMerger"),
                        0
                    )
                    insert_idx = merger_idx + 1

                config["pipeline"].insert(insert_idx, {
                    "type": "DefaultValueFilter",
                    "config": default_config
                })

        # Sample FormattingConsolidator (inserted after DefaultValueFilter, before sequencer)
        if "FormattingConsolidator" in self.distributions:
            fmt_dist = self.distributions["FormattingConsolidator"]
            fmt_config = {}

            for param, options in fmt_dist.items():
                fmt_config[param] = self._weighted_choice(options)

            if fmt_config.get("enabled", True):
                # Insert before OperationSequencer
                sequencer_idx = next(
                    (i for i, t in enumerate(config["pipeline"]) if t["type"] == "OperationSequencer"),
                    len(config["pipeline"])
                )
                config["pipeline"].insert(sequencer_idx, {
                    "type": "FormattingConsolidator",
                    "config": fmt_config
                })

        # Sample InputSplitter (inserted after OperationMerger, before OperationSequencer)
        if "InputSplitter" in self.distributions:
            is_dist = self.distributions["InputSplitter"]
            is_config = {}

            for param, options in is_dist.items():
                is_config[param] = self._weighted_choice(options)

            if is_config.get("enabled", True):
                # Insert right before OperationSequencer
                sequencer_idx = next(
                    (i for i, t in enumerate(config["pipeline"])
                     if t["type"] == "OperationSequencer"),
                    len(config["pipeline"])
                )
                config["pipeline"].insert(sequencer_idx, {
                    "type": "InputSplitter",
                    "config": is_config
                })

        # Sample AutoFillDetector (inserted after OperationSequencer)
        if "AutoFillDetector" in self.distributions:
            af_dist = self.distributions["AutoFillDetector"]
            af_config = {}

            for param, options in af_dist.items():
                af_config[param] = self._weighted_choice(options)

            if af_config.get("enabled", True):
                sequencer_idx = next(
                    (i for i, t in enumerate(config["pipeline"])
                     if t["type"] == "OperationSequencer"),
                    None
                )
                insert_idx = (sequencer_idx + 1) if sequencer_idx is not None else len(config["pipeline"])
                config["pipeline"].insert(insert_idx, {
                    "type": "AutoFillDetector",
                    "config": af_config
                })

        return config


def sample_config(
    distributions: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convenience function to sample a configuration.

    Args:
        distributions: Custom probability distributions. If None, uses defaults.
        seed: Random seed for reproducibility.

    Returns:
        A sampled configuration dict for SequencingEngine.

    Example:
        >>> config = sample_config(seed=42)
        >>> from next_action_pred_eval.generation.sequencing import SequencingEngine
        >>> engine = SequencingEngine(config)
    """
    sampler = ConfigSampler(distributions=distributions, seed=seed)
    return sampler.sample()
