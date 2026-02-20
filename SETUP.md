# Environment Setup (Reproducible)

> These instructions get the full MARL training environment running on a
> fresh Linux machine (e.g. a cloud GPU instance).

## 1. Clone the repo

```bash
git clone <your-repo-url> memeplex-capstone
cd memeplex-capstone
```

## 2. Install Python deps for SMACv2 MAPPO

> **Note:** `agilerl` (in `requirements.txt`) pulls in `vllm==0.10.0` which has
> no wheel for Python 3.13. Install only what `run_smacv2_mappo.py` actually
> needs:

```bash
# Clear pip cache first if disk is tight (~1.7 GB freed)
pip cache purge

# Install CPU-only torch (saves ~2 GB vs CUDA build) + SMAC v2 stack
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy tqdm matplotlib "gymnasium>=0.28.0"
pip install "smacv2 @ git+https://github.com/oxwhirl/smacv2.git@main"
# Fix dateutil/six conflict introduced by smacv2's enum34 dep
pip install six --upgrade
```

## 3. Install StarCraft II + SMAC v2 maps

```bash
bash install_sc2.sh
```

The script downloads StarCraft II Linux client v4.10 (~4 GB) and the
SMAC v2 map files. By default it installs to `~/StarCraftII`.

## 4. Fix SC2PATH — /dev/shm noexec workaround

> **Important (containerised environments):** If SC2 was installed into
> `/dev/shm/StarCraftII` (e.g. a Docker container with `shm` mounted
> `noexec`), the SC2 binary cannot execute from there even though mode bits
> look correct. Copy the binary-containing directories to a normal filesystem:

```bash
# Only copy the executables (~640 MB); symlink the large data dirs
mkdir -p /workspace/StarCraftII
cp -r /dev/shm/StarCraftII/Versions /workspace/StarCraftII/
cp -r /dev/shm/StarCraftII/Libs     /workspace/StarCraftII/
cp -r /dev/shm/StarCraftII/Interfaces /workspace/StarCraftII/
cp    /dev/shm/StarCraftII/.build.info /workspace/StarCraftII/
ln -sfn /dev/shm/StarCraftII/SC2Data   /workspace/StarCraftII/SC2Data
ln -sfn /dev/shm/StarCraftII/Maps      /workspace/StarCraftII/Maps
ln -sfn /dev/shm/StarCraftII/Battle.net /workspace/StarCraftII/Battle.net
ln -sfn /dev/shm/StarCraftII/Replays   /workspace/StarCraftII/Replays

export SC2PATH=/workspace/StarCraftII
```

Add `export SC2PATH=/workspace/StarCraftII` to `~/.bashrc` to persist it.

> If SC2 installed normally to `~/StarCraftII` (no `/dev/shm`), just:
> ```bash
> export SC2PATH="$HOME/StarCraftII"
> ```

## 5. Quick smoke test

> **Note:** `python` on this system is Python 3.8 (no packages). Always use
> `python3.13` (where pip installed everything).

```bash
export SC2PATH=/workspace/StarCraftII   # if not already in .bashrc

# Verify imports
python3.13 -c "from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper; print('smacv2 OK')"

# Run a single episode
python3.13 run_smacv2_mappo.py --mode test --test-episodes 1
```

Expected output ends with `Smoke test passed!`.

## 6. Training

```bash
# MAPPO training on SMAC v2 (terran 5v5, 2M steps default)
python3.13 run_smacv2_mappo.py --mode train

# Shorter run for quick iteration
python3.13 run_smacv2_mappo.py --mode train --total-steps 200000

# Different scenario
python3.13 run_smacv2_mappo.py --mode train --race protoss --n-units 10 --n-enemies 10

# See all options
python3.13 run_smacv2_mappo.py --help
```

## Summary of what's needed

| Component | How | Size |
|-----------|-----|------|
| Python deps (SMAC only) | `pip install torch numpy tqdm matplotlib gymnasium smacv2` | ~1 GB (CPU torch) |
| StarCraft II | `bash install_sc2.sh` | ~4 GB |
| SMAC v2 maps | included in `install_sc2.sh` | ~177 MB |
| SC2 binaries in exec-safe path | copy `Versions/` + `Libs/` to `/workspace` | ~640 MB |
