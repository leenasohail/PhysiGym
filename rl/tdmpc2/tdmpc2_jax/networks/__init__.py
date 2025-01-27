import sys, os
absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("PhysiCell") + len("PhysiCell")
]
sys.path.append(absolute_path)

from rl.tdmpc2.tdmpc2_jax.networks.ensemble import Ensemble
from rl.tdmpc2.tdmpc2_jax.networks.mlp import NormedLinear
