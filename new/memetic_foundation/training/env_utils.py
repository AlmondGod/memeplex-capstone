"""
env_utils.py — SMACv2 environment helpers.

Provides map configs, distribution configs, and environment factory
for StarCraft Multi-Agent Challenge v2.
"""

from __future__ import annotations

import os

# Mac: default SC2PATH to Blizzard app location if not set
if not os.environ.get("SC2PATH"):
    _mac_default = "/Applications/StarCraft II"
    if os.path.isdir(_mac_default):
        os.environ["SC2PATH"] = _mac_default


RACE_CONFIGS = {
    "terran": {
        "unit_types": ["marine", "marauder", "medivac"],
        "exception_unit_types": ["medivac"],
        "weights": [0.45, 0.45, 0.1],
    },
    "protoss": {
        "unit_types": ["stalker", "zealot", "colossus"],
        "exception_unit_types": ["colossus"],
        "weights": [0.45, 0.45, 0.1],
    },
    "zerg": {
        "unit_types": ["zergling", "baneling", "hydralisk"],
        "exception_unit_types": ["baneling"],
        "weights": [0.45, 0.1, 0.45],
    },
}

RACE_MAP_NAMES = {
    "terran": "10gen_terran",
    "protoss": "10gen_protoss",
    "zerg": "10gen_zerg",
}


def build_distribution_config(race: str, n_units: int, n_enemies: int) -> dict:
    """Build the SMACv2 capability/distribution config."""
    race = race.lower()
    if race not in RACE_CONFIGS:
        raise ValueError(f"Unknown race '{race}'. Choose from: {list(RACE_CONFIGS)}")
    rc = RACE_CONFIGS[race]
    return {
        "n_units": n_units,
        "n_enemies": n_enemies,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": rc["unit_types"],
            "exception_unit_types": rc["exception_unit_types"],
            "weights": rc["weights"],
            "observe": True,
        },
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "n_enemies": n_enemies,
            "map_x": 32,
            "map_y": 32,
        },
    }


def make_env(race: str, n_units: int, n_enemies: int, render: bool = False):
    """Create a single SMACv2 environment instance."""
    from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

    dist_config = build_distribution_config(race, n_units, n_enemies)
    window_x, window_y = (1280, 720) if render else (640, 480)
    env = StarCraftCapabilityEnvWrapper(
        capability_config=dist_config,
        map_name=RACE_MAP_NAMES[race],
        debug=False,
        conic_fov=False,
        obs_own_pos=True,
        use_unit_ranges=True,
        min_attack_range=2,
        window_size_x=window_x,
        window_size_y=window_y,
    )
    return env
