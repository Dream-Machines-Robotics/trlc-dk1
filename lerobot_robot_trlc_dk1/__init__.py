# from .leader import DK1Leader, DK1LeaderConfig
# from .follower import DK1Follower, DK1FollowerConfig
# from .bi_leader import BiDK1Leader, BiDK1LeaderConfig
# from .bi_follower import BiDK1Follower, BiDK1FollowerConfig

# __all__ = ["DK1Leader", "DK1LeaderConfig", "DK1Follower", "DK1FollowerConfig", "BiDK1Leader", "BiDK1LeaderConfig", "BiDK1Follower", "BiDK1FollowerConfig"]

from __future__ import annotations

__all__ = [
    "DK1Leader", "DK1LeaderConfig",
    "DK1Follower", "DK1FollowerConfig",
    "BiDK1Leader", "BiDK1LeaderConfig",
    "BiDK1Follower", "BiDK1FollowerConfig",
]

def __getattr__(name: str):
    if name in ("DK1Leader", "DK1LeaderConfig"):
        from .leader import DK1Leader, DK1LeaderConfig
        return DK1Leader if name == "DK1Leader" else DK1LeaderConfig
    if name in ("DK1Follower", "DK1FollowerConfig"):
        from .follower import DK1Follower, DK1FollowerConfig
        return DK1Follower if name == "DK1Follower" else DK1FollowerConfig
    if name in ("BiDK1Leader", "BiDK1LeaderConfig"):
        from .bi_leader import BiDK1Leader, BiDK1LeaderConfig
        return BiDK1Leader if name == "BiDK1Leader" else BiDK1LeaderConfig
    if name in ("BiDK1Follower", "BiDK1FollowerConfig"):
        from .bi_follower import BiDK1Follower, BiDK1FollowerConfig
        return BiDK1Follower if name == "BiDK1Follower" else BiDK1FollowerConfig
    raise AttributeError(name)