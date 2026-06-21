from typing import Optional
from pathlib import Path

from quantaalpha.coder.costeer.config import CoSTEERSettings
from quantaalpha.core.conf import ExtendedSettingsConfigDict


class FactorCoSTEERSettings(CoSTEERSettings):
    model_config = ExtendedSettingsConfigDict(env_prefix="FACTOR_CoSTEER_")

    data_folder: str = "git_ignore_folder/factor_implementation_source_data"
    """Path to the folder containing financial data (default is fundamental data in Qlib)"""

    data_folder_debug: str = "git_ignore_folder/factor_implementation_source_data_debug"
    """Path to the folder containing partial financial data (for debugging)"""

    simple_background: bool = True
    """Whether to use simple background information for code feedback"""

    file_based_execution_timeout: int = 1200
    """Timeout in seconds for each factor implementation execution"""

    select_method: str = "random"
    """Method for the selection of factors implementation"""

    python_bin: str = "python"
    """Path to the Python binary"""
    
    factor_zoo_path: Optional[str] = str(Path(__file__).resolve().parents[1] / "factor_zoo" / "default_zoo.csv")
    """Path to the CSV file containing the factor zoo database (e.g., Alpha101 factors).
    Dedup / novelty check (duplication detection) is ON by default and points to the
    bundled default_zoo.csv. Set to None to disable the factor-zoo duplication check;
    in that case only free arguments ratio and unique variables ratio checks are performed."""
    
    duplication_threshold: int = 8
    """Threshold for duplication detection. If duplicated subtree size exceeds this value, 
    the factor will be rejected."""

    symbol_length_threshold: int = 300
    """Maximum allowed symbol length (SL) for factor expressions. 
    Expressions longer than this threshold will be rejected to prevent overfitting."""
    
    base_features_threshold: int = 6
    """Maximum allowed number of unique base features (ER) in factor expressions.
    Base features are raw variables like $close, $open, $high, $low, $volume.
    Expressions using more than this number of distinct base features will be rejected."""

    max_nodes_threshold: int = 60
    """Hard cap on the total number of AST nodes in a factor expression.
    Expressions with more than this many nodes are rejected outright."""

    max_depth_threshold: int = 8
    """Hard cap on the maximum AST depth of a factor expression.
    Expressions deeper than this are rejected outright."""


FACTOR_COSTEER_SETTINGS = FactorCoSTEERSettings()
