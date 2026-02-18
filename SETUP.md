# Environment Setup (Reproducible)

> These instructions get the full MARL training environment running on a
> fresh Linux machine (e.g. a cloud GPU instance).

## 1. Clone the repo

```bash
git clone <your-repo-url> memeplex-capstone
cd memeplex-capstone
```

## 2. Create a virtual environment & install Python deps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

This installs **all** Python dependencies including:
- The existing MPE/PettingZoo/AgileRL stack
- **SMACv2** (from the [oxwhirl/smacv2](https://github.com/oxwhirl/smacv2) repo)

## 3. Install StarCraft II + SMAC v2 maps

```bash
# Optional: set install location (default: ~/StarCraftII)
# export SC2PATH=/path/to/StarCraftII

bash install_sc2.sh
```

The script downloads StarCraft II Linux client v4.10 (~4 GB) and the
SMAC v2 map files (`32x32_flat.SC2Map`, etc.).

## 4. Set the SC2PATH environment variable

Add this to your shell profile (`.bashrc` / `.zshrc`):

```bash
export SC2PATH="$HOME/StarCraftII"
```

Then `source ~/.bashrc` (or reopen a terminal).

## 5. Quick smoke test

```bash
# Verify smacv2 Python package imports correctly
python -c "from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper; print('smacv2 OK')"

# Run a short random-agent episode (requires StarCraft II)
python run_smacv2_mappo.py --mode test --test-episodes 1
```

## 6. Training

```bash
# MAPPO training on SMAC v2 (terran 5v5 default)
python run_smacv2_mappo.py --mode train

# Different scenario
python run_smacv2_mappo.py --mode train --race protoss --n-units 10 --n-enemies 10

# See all options
python run_smacv2_mappo.py --help
```

## Summary of what's needed

| Component | How | Size |
|-----------|-----|------|
| Python deps | `pip install -r requirements.txt` | ~1 GB |
| StarCraft II | `bash install_sc2.sh` | ~4 GB |
| SMAC v2 maps | (included in install_sc2.sh) | ~500 KB |
