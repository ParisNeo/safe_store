"""Shared model server sub-package for safe_store.

Provides the background model server daemon, its client, and the wire
protocol used by STVectorizer's shared-server mode.

This file MUST exist so that hatchling treats ``model_server`` as a Python
package and bundles it into the wheel. Without it, ``python -m
safe_store.vectorization.model_server.run_server`` fails with
``No module named ...`` on installed wheels.
"""