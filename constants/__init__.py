from pathlib import Path
from dataclasses import dataclass
from typing import List
import math


@dataclass
class CompetitionParameters:
    """Class defining model parameters for a competition."""
    
    reward_percentage: float  # Reward percentage for the competition
    competition_id: str  # Unique identifier for the competition


# ---------------------------------
# Project Constants
# ---------------------------------

WANDB_PROJECT: str = "myshell-tts-subnet"  # WANDB project name
SUBNET_UID: int = 3  # Unique ID for the subnet
SUBNET_START_BLOCK: int = 2635801  # Start block number for the subnet
ROOT_DIR: Path = Path(__file__).parent.parent  # Root directory of the project
MAX_HUGGING_FACE_BYTES: int = 512 * 1024 * 1024  # Maximum size in bytes for Hugging Face repository
ORIGINAL_COMPETITION_ID: str = "p240"  # Default competition ID
CONSTANT_ALPHA: float = 0.2  # Alpha value for enhancing trust
TIMESTAMP_EPSILON: float = 0.04  # Epsilon value for enhancing trust

# Competition schedule containing model architectures
COMPETITION_SCHEDULE: List[CompetitionParameters] = [
    CompetitionParameters(
        reward_percentage=1.0,
        competition_id=ORIGINAL_COMPETITION_ID,
    ),
]

# Ensure that the total reward percentage sums up to 1.0
assert math.isclose(sum(x.reward_percentage for x in COMPETITION_SCHEDULE), 1.0), "Total reward percentage must equal 1.0"

# ---------------------------------
# Miner/Validator Model Parameters
# ---------------------------------

WEIGHTS_VERSION_KEY: int = 4  # Version key for weights

# Validator model parameters
LR: float = 0.2  # Learning rate, alpha = 1 - LR
TEMPERATURE: float = 0.08  # Scoring exponential temperature for validator
