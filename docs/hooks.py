"""MkDocs event hooks — suppress known third-party deprecation warnings."""
import warnings


def on_startup(**_kwargs):
    # jieba uses pkg_resources (deprecated in setuptools >= 81)
    warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
    warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*pkg_resources.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*declare_namespace.*")
