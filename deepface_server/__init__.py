"""DeepFace Age Estimator package."""
__version__ = "0.3.0"

from .app_factory import create_app  # noqa: E402

__all__ = ["create_app", "__version__"]