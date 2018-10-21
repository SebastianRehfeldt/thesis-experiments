from .config_algorithms_comp import ALGORITHMS as COMPETITORS
from .config_algorithms_rar import ALGORITHMS as RAR
from .config_algorithms_rar_as import ALGORITHMS as RAR_AS
from .config_algorithms_rar_del import ALGORITHMS as RAR_DEL
from .config_algorithms_rar_cache import ALGORITHMS as RAR_CACHE
from .config_algorithms_rar_imp import ALGORITHMS as RAR_IMP
from .config_algorithms_rar_boost import ALGORITHMS as RAR_BOOST
from .config_dataset_synthetic import DATASET_CONFIG as SYNTHETIC_CONFIG
from .config_dataset_uci import DATASET_CONFIG as UCI_CONFIG
from .config_runs import CONFIG

__all__ = [
    "COMPETITORS",
    "RAR",
    "SYNTHETIC_CONFIG",
    "CONFIG",
    "UCI_CONFIG",
    "RAR_AS",
    "RAR_DEL",
    "RAR_CACHE",
    "RAR_IMP",
    "RAR_BOOST",
]
