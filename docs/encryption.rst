==========
Encryption
==========

``safe_store`` provides optional encryption at rest for the text content of document chunks stored in the database. This helps protect sensitive information if the database file itself is exposed.

How it Works
------------

*   **Algorithm:** Uses Fernet symmetric authenticated cryptography from the `cryptography <https://cryptography.io/en/latest/fernet/>`_ library. Fernet uses AES-128 in CBC mode with PKCS7 padding for encryption and HMAC with SHA256 for authentication.
*   **Key Derivation:** When you provide an ``encryption_key`` (password) during ``safe_store`` initialization, a strong 256-bit encryption key suitable for Fernet is derived using PBKDF2 HMAC SHA256.
    *   **Salt:** For simplicity within ``safe_store``, a **fixed, hardcoded salt** is used during key derivation. This means the same password will always produce the same encryption key. See the Security Considerations below.
*   **Encryption Target:** Only the ``chunk_text`` stored in the ``chunks`` table is encrypted. Other data like document paths, metadata, vectorizer parameters, and the vectors themselves are **not** encrypted by this feature.
*   **Automatic Handling:** Encryption and decryption are handled automatically during ``add_document`` and ``query`` operations if the ``safe_store`` instance was initialized with the correct ``encryption_key``.

Enabling Encryption
-------------------

1.  **Install Dependency:** Ensure the ``cryptography`` library is installed:
    .. code-block:: bash

       pip install safe_store[encryption]
       # or
       pip install safe_store[all]

2.  **Provide Key on Init:** Pass your chosen password (key) to the ``encryption_key`` parameter when creating the ``safe_store`` instance:

    .. code-block:: python

       import safe_store

       my_password = "your-very-strong-password-here" # Keep this safe!

       store = safe_store.safe_store(
           "encrypted_store.db",
           encryption_key=my_password
       )

       # Now, when you add documents, chunk text will be encrypted
       with store:
           store.add_document("path/to/sensitive_doc.txt")

           # When you query, chunk text will be automatically decrypted
           results = store.query("search term")
           print(results[0]['chunk_text']) # Prints decrypted text

Usage Notes
-----------

*   **Consistency:** You **must** use the exact same ``encryption_key`` every time you open a specific database file that contains encrypted data.
*   **Querying without Key:** If you open an encrypted database without providing the key (or with the wrong key), query results will contain placeholder text like ``[Encrypted - Key Unavailable]`` or ``[Encrypted - Decryption Failed]`` instead of the actual chunk text.
*   **Adding Vectorizations:** If you use ``add_vectorization`` for a method like TF-IDF that requires fitting on existing text, ``safe_store`` will attempt to decrypt the necessary chunks using the provided key. If the key is missing or incorrect, the operation will fail.
*   **Key Management:** **You are solely responsible for managing your ``encryption_key`` securely.** If you lose the key, the encrypted data in the database will be permanently unrecoverable. Do not hardcode keys directly in your source code in production environments. Consider using environment variables, configuration files with appropriate permissions, or dedicated secrets management systems.

Security Considerations
-----------------------

*   **Fixed Salt:** As mentioned, ``safe_store`` currently uses a fixed salt for PBKDF2 key derivation for simplicity. This is less secure than using a unique, randomly generated salt for each password/database, as it doesn't fully protect against precomputed rainbow table attacks if the fixed salt becomes known. For high-security requirements, this implementation might not be sufficient.
*   **Metadata Not Encrypted:** Document paths, metadata, and vector information remain unencrypted. Ensure no sensitive information is placed in document metadata if the database file requires protection.
*   **Scope:** Encryption only applies to chunk text *at rest* in the SQLite file. Data is decrypted in memory during processing (e.g., querying).

This feature provides a reasonable layer of protection against casual inspection of the database file but relies heavily on the security of your chosen ``encryption_key`` and understanding its limitations.
