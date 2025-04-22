============
Installation
============

Install ``safe_store`` using pip:

.. code-block:: bash

   pip install safe_store

Optional Dependencies
---------------------

``safe_store`` uses optional dependencies for certain features like specific vectorizers or document parsers. You can install these extras as needed:

*   **Sentence Transformers:** For state-of-the-art sentence embeddings.
    .. code-block:: bash

       pip install safe_store[sentence-transformers]

*   **TF-IDF:** For classic TF-IDF vectorization (requires scikit-learn).
    .. code-block:: bash

       pip install safe_store[tfidf]

*   **Document Parsing:** For handling ``.pdf``, ``.docx``, and ``.html`` files.
    .. code-block:: bash

       pip install safe_store[parsing]

*   **Encryption:** For encrypting chunk text at rest (requires cryptography).
    .. code-block:: bash

       pip install safe_store[encryption]

*   **All Features:** To install all optional dependencies at once.
    .. code-block:: bash

       pip install safe_store[all]

*   **Development:** To install dependencies needed for testing, building, and documentation generation.
    .. code-block:: bash

       pip install safe_store[dev]
