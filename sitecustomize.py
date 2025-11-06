"""
Site customization module for NodeRAG.
This module is automatically imported by Python on startup.
"""
import warnings
import sys

# Suppress specific RuntimeError warnings from asyncio event loop closure
# This prevents noisy error messages when httpx clients cleanup after event loop closes
def custom_excepthook(exc_type, exc_value, exc_traceback):
    """Custom exception hook to suppress specific asyncio cleanup errors."""
    # Suppress "Event loop is closed" errors from background tasks
    if exc_type is RuntimeError and "Event loop is closed" in str(exc_value):
        return
    # Call the default exception handler for all other exceptions
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

# Install custom exception hook
sys.excepthook = custom_excepthook

# Also suppress warnings about coroutines not being awaited in cleanup
warnings.filterwarnings("ignore", message=".*Event loop is closed.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine.*was never awaited.*")
