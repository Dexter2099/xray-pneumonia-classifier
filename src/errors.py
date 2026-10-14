"""Recoverable application failures, independent of the user interface."""


class XrayError(Exception):
    """A failure that an application adapter can explain to its user."""


class ImageInputError(XrayError, ValueError):
    """An image or tensor does not satisfy the model's input contract."""


class ConfigurationError(XrayError, ValueError):
    """Application settings are missing or incompatible with the checkpoint."""


class ModelLoadError(XrayError):
    """The configured model checkpoint cannot be loaded safely."""


class PredictionError(XrayError):
    """Inference failed or produced an invalid result."""


class ExplanationError(XrayError):
    """No usable Grad-CAM explanation could be generated."""
