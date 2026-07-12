"""JCRec with CLIRS train/test split — fair baseline (primary compare vs CLIRS)."""

from JCRecFair.Reinforce import JcrecFairReinforce
from JCRecFair.split_sync import (
    ClirsSplitNotFoundError,
    ensure_clirs_split,
    publish_clirs_split_artifact,
    require_clirs_split_file,
)

__all__ = (
    "JcrecFairReinforce",
    "ClirsSplitNotFoundError",
    "ensure_clirs_split",
    "publish_clirs_split_artifact",
    "require_clirs_split_file",
)
