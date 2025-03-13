"""
Utility to disable JAX debug messages.
Import this module at the beginning of any script that uses JAX to disable debug messages.
"""

import logging


def disable_jax_logging():
    """
    Disable JAX debug messages by setting the logging level for JAX loggers to ERROR.
    """
    # Disable specific JAX loggers that produce debug messages
    logging.getLogger("jax._src.cache_key").setLevel(logging.ERROR)
    logging.getLogger("jax._src.lib.xla_bridge").setLevel(logging.ERROR)
    logging.getLogger("jax._src.lib.xla_client").setLevel(logging.ERROR)
    logging.getLogger("jax._src.interpreters").setLevel(logging.ERROR)
    logging.getLogger("jax._src.dispatch").setLevel(logging.ERROR)

    # Disable all JAX loggers (more comprehensive approach)
    logging.getLogger("jax").setLevel(logging.ERROR)


# Automatically disable JAX logging when this module is imported
disable_jax_logging()
