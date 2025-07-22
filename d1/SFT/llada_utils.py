import importlib.util, sys, types, pathlib
from huggingface_hub import hf_hub_download
from typing import Tuple, Type
import tempfile


# ------------------------------------------------------------
# load llada modules
# ------------------------------------------------------------

def load_llada_modules(
    repo_id: str = "GSAI-ML/LLaDA-8B-Base",
) -> Tuple[Type, ...]:
    """
    Returns:
        (LLaDASequentialBlock, ModelConfig, ActivationType, LayerNormType)
    """
    tmp_dir = tempfile.mkdtemp(prefix="llada_")
    modeling_path      = hf_hub_download(repo_id, "modeling_llada.py",
                                         local_dir=tmp_dir, local_dir_use_symlinks=False)
    configuration_path = hf_hub_download(repo_id, "configuration_llada.py",
                                         local_dir=tmp_dir, local_dir_use_symlinks=False)

    pkg_name = "llada_remote"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [tmp_dir]
    sys.modules[pkg_name] = pkg

    spec_conf = importlib.util.spec_from_file_location(
        f"{pkg_name}.configuration_llada", configuration_path)
    conf_mod = importlib.util.module_from_spec(spec_conf)
    sys.modules[spec_conf.name] = conf_mod
    spec_conf.loader.exec_module(conf_mod)

    spec_model = importlib.util.spec_from_file_location(
        f"{pkg_name}.modeling_llada", modeling_path)
    model_mod = importlib.util.module_from_spec(spec_model)
    sys.modules[spec_model.name] = model_mod
    spec_model.loader.exec_module(model_mod) # type: ignore[arg-type]

    return (
        model_mod.LLaDASequentialBlock,
        model_mod.ModelConfig,
        model_mod.ActivationType,
        model_mod.LayerNormType,
    )