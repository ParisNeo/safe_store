.. safe_store documentation master file, created by
   sphinx-quickstart on <date>.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to safe_store's documentation!
=====================================

**safe_store** is a Python library providing a lightweight, file-based vector database using SQLite. It's designed for simplicity and efficiency, making it ideal for integrating into local Retrieval-Augmented Generation (RAG) pipelines.

Key Features:

*   **Local SQLite Backend:** Simple, single-file database.
*   **Concurrency Safe:** Handles multiple processes writing via file locks.
*   **Multiple Vectorizers:** Supports Sentence Transformers, TF-IDF, etc.
*   **Document Parsing:** Handles `.txt`, `.pdf`, `.docx`, `.html`.
*   **Optional Encryption:** Securely store chunk text at rest.
*   **Informative Logging:** Clear console output via `ascii_colors`.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   logging
   encryption
   graph_store
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
