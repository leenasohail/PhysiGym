import sys, os
absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("PhysiGym_Models") + len("PhysiGym_Models")
]
sys.path.append(absolute_path)
from rl.tdmpc2.tdmpc2_jax.tdmpc2 import TDMPC2
from rl.tdmpc2.tdmpc2_jax.world_model import WorldModel