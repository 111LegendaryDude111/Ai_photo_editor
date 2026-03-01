class PipelineError(Exception):
    """Base error for pipeline failures."""


class NoFaceDetectedError(PipelineError):
    """Raised when no reliable face is found in an image."""


class InvalidMaskError(PipelineError):
    """Raised when mask payload cannot be parsed or validated."""


class MissingLoraError(PipelineError):
    """Raised when requested LoRA checkpoint cannot be loaded."""


class InputValidationError(PipelineError):
    """Raised when user input image/payload violates policy limits."""


class GpuOutOfMemoryError(PipelineError):
    """Raised when generation fails because of GPU OOM."""
