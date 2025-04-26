mkdir Physi
cd Physi
git clone https://github.com/MathCancer/PhysiCell.git
git clone https://github.com/Dante-Berth/PhysiGym.git
python3 -m venv .venv
source .venv/bin/activate
cd PhysiGym
python3 install_rl_folder.py
python3 install_physigym.py complex_tme --force
cd ../PhysiCell
pip install -r rl/sac/sac_requirements.txt
make load PROJ=physigym_tme
make
