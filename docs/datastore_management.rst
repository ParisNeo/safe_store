Datastore Management
====================

This guide covers the day-to-day operations on a ``SafeStore`` database after
it has been created: renaming the datastore, editing its description and
metadata, tuning chunking parameters, and switching or re-embedding with a
different vectorizer.

All operations below are performed on a live ``SafeStore`` instance. The
library persists changes atomically and updates the underlying SQLite
``store_metadata`` table so they survive process restarts.

.. contents:: On this page
   :local:
   :depth: 2


.. rubric:: Quick reference

=============================================  =============================================
Task                                           API
=============================================  =============================================
Rename the datastore                           ``store.update_properties({"name": ...})``
Edit description                               ``store.update_properties({"description": ...})``
Edit arbitrary metadata                        ``store.update_properties(metadata={...})``
Inspect current properties                     ``store.get_properties()``
Full diagnostics                               ``store.info()``
Change chunk size / overlap / strategy         ``store.rechunk_and_vectorize(...)``
Change only the embedding model                ``store.revectorize_database(...)``
Migrate everything (chunks + embeddings)       ``store.rechunk_and_vectorize(...)``
=============================================  =============================================


Opening a datastore
-------------------

Rename, description, and parameter changes are performed on **an existing
datastore**, so it must first be opened. Two constructors matter here:

.. code-block:: python

   import safe_store

   # Option A — strict: raises ConfigurationError on vectorizer mismatch.
   #           Use this when you know exactly what vectorizer the DB uses.
   store = safe_store.SafeStore("knowledge.db")

   # Option B — lenient: reopens the DB using its *own* recorded vectorizer.
   #           Use this for inspection tools / generic panels that must not
   #           fail on a mismatch (see § "Reopening a datastore whose
   #           vectorizer differs" at the bottom).
   store = safe_store.SafeStore.from_db_auto("knowledge.db")

Because ``SafeStore`` is also a context manager, the recommended pattern is:

.. code-block:: python

   with safe_store.SafeStore.from_db_auto("knowledge.db") as store:
       # do work — the connection is closed cleanly on exit
       ...


Renaming the datastore
----------------------

The **name** is a human-readable label stored in ``store_metadata.store_name``.
It is *not* the file name — renaming the datastore does not renaming the
``.db`` file.

.. code-block:: python

   with safe_store.SafeStore.from_db_auto("knowledge.db") as store:
       # Read current properties first (recommended before mutating)
       props = store.get_properties()
       print(f"Current name: {props['name']!r}")

       # Rename
       store.update_properties({"name": "Enterprise Knowledge Base"})

       assert store.get_properties()["name"] == "Enterprise Knowledge Base"

.. note::

   Renaming the **file** on disk is a separate, entirely filesystem operation.
   ``SafeStore`` stores the absolute path resolved at open time, so the file
   must be renamed **while the store is closed**:

   .. code-block:: python

      import shutil, pathlib

      src = pathlib.Path("knowledge.db")
      dst = pathlib.Path("enterprise_kb.db")

      # Make sure no SafeStore instance holds the file open.
      for suffix in ("", "-wal", "-shm", ".lock"):
          s = src.with_name(src.name + suffix)
          if s.exists():
              shutil.move(str(s), str(dst.with_name(dst.name + suffix)))


Editing the description
-----------------------

The **description** is stored in ``store_metadata.store_description``. It is a
free-form single-line string intended for a short summary — it appears in the
Studio hub cards and in ``store.info()`` output.

.. code-block:: python

   with safe_store.SafeStore.from_db_auto("knowledge.db") as store:
       store.update_properties({
           "description": "Internal engineering docs, incident reports, and architecture RFCs (2023-2026)."
       })

       print(store.get_properties()["description"])

To clear the description, set it to ``None`` or an empty string:

.. code-block:: python

   store.update_properties({"description": None})


Editing arbitrary metadata
--------------------------

Beyond ``name`` and ``description``, ``SafeStore`` stores an arbitrary
``metadata`` dict (JSON-serialised) under ``store_metadata.store_metadata``.
By default, ``update_properties(metadata=...)`` **deep-merges** into the
existing dict; pass ``overwrite_metadata=True`` to replace it wholesale.

.. code-block:: python

   with safe_store.SafeStore.from_db_auto("knowledge.db") as store:
       # Deep-merge — existing keys are preserved
       store.update_properties(metadata={
           "owner": "Platform Team",
           "classification": "internal",
           "tags": ["rag", "engineering"],
       })

       # Wholesale replacement — previous metadata is discarded
       store.update_properties(
           metadata={"owner": "SRE Team"},
           overwrite_metadata=True,
       )

       print(store.get_properties()["metadata"])

.. tip::

   The ``ontology`` key under ``metadata`` is special — it is consumed by
   :class:`GraphStore` when building the knowledge graph. Keep it intact
   unless you have a reason to replace it.


One-shot update helper
~~~~~~~~~~~~~~~~~~~~~~

All of ``name``, ``description`` and ``metadata`` can be updated in a single
atomic call:

.. code-block:: python

   store.update_properties(
       properties={
           "name": "Enterprise KB",
           "description": "Architecture + SRE docs",
       },
       metadata={"owner": "Platform Team"},
   )


Inspecting current configuration
--------------------------------

Before tuning parameters, read what the datastore currently thinks its
configuration is.

.. code-block:: python

   with safe_store.SafeStore.from_db_auto("knowledge.db") as store:
       props = store.get_properties()
       print(f"name         : {props['name']!r}")
       print(f"description  : {props['description']!r}")
       print(f"vectorizer   : {props['vectorizer_name']}:{props['vectorizer_config']}")
       print(f"chunk_size   : {props['chunk_size']}")
       print(f"chunk_overlap: {props['chunk_overlap']}")
       print(f"strategy     : {props['chunking_strategy']}")

For the full diagnostic view (document counts, per-document chunk counts,
graph topology, ontology, encryption state):

.. code-block:: python

   store.info()          # prints a formatted ASCII panel
   diag = store.get_database_info(print_summary=False)


Changing chunking parameters
----------------------------

There are two distinct knobs, and it matters which one is used:

- ``chunk_size`` / ``chunk_overlap`` / ``chunking_strategy`` **affect only
  newly ingested documents**. Changing them does **not** re-chunk documents
  that are already in the datastore.
- To **rebuild** every document under a new chunking configuration, use
  ``rechunk_and_vectorize(...)`` (see below).

Setting defaults for future documents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass the parameters at ``SafeStore`` construction; they are persisted to
``store_metadata.store_config`` and used on every subsequent ``add_document``
or ``add_text`` call:

.. code-block:: python

   store = safe_store.SafeStore(
       db_path="knowledge.db",
       chunk_size=1024,
       chunk_overlap=128,
       chunking_strategy="recursive",
   )
   # Reopen once — the new values are now the persisted defaults.
   store.close()

   # Verify
   with safe_store.SafeStore.from_db_auto("knowledge.db") as s:
       assert s.chunk_size == 1024
       assert s.chunk_overlap == 128
       assert s.chunking_strategy == "recursive"

Available ``chunking_strategy`` values: ``character``, ``token``, ``paragraph``,
``semantic``, ``recursive``, ``structure``, ``markdown``, ``contextual``, ``late``.

.. warning::

   ``chunk_overlap`` must be **strictly less than** ``chunk_size``. ``SafeStore``
   raises :class:`ConfigurationError` at construction otherwise.

Rebuilding every document under new chunking
--------------------------------------------

``store.rechunk_and_vectorize(...)`` reconstructs every document from the
datastore, re-chunks it with the supplied parameters, and re-embeds the new
chunks — **atomically, per document**. Reconstruction happens *before* any
delete, so a mid-run failure cannot lose data.

.. code-block:: python

   with safe_store.SafeStore.from_db_auto("knowledge.db") as store:
       summary = store.rechunk_and_vectorize(
           new_chunk_size=1024,
           new_chunk_overlap=128,
           new_chunking_strategy="recursive",
       )
       print(summary)
       # {'documents_processed': 42, 'documents_failed': 0,
       #  'chunks_created': 1873, 'chunk_size': 1024, 'chunk_overlap': 128,
       #  'chunking_strategy': 'recursive', 'vectorizer_name': 'st'}

Every parameter is optional; unspecified ones fall back to the current value.
**At least one parameter must be supplied**, otherwise ``ConfigurationError``
is raised (it would be a no-op that needlessly re-embeds the entire database).

.. important::

   ``rechunk_and_vectorize`` is **destructive of chunk IDs**. Chunk IDs are
   re-assigned; if you have external systems that store legacy ``chunk_id``
   values (e.g. ``node_chunk_links`` beyond what GraphStore manages), back up
   with ``store.export_db_file("backup.db")`` first.


Switching the vectorizer
------------------------

There are three migration tiers, ordered from least to most expensive:

=================================  =================================  ==================
Scenario                           Method                             Preserves
=================================  =================================  ==================
Change only the model name         ``revectorize_database(...)``      chunk boundaries
Change only the vectorizer family  ``revectorize_database(...)``      chunk boundaries
Change chunks AND embeddings       ``rechunk_and_vectorize(...)``     nothing (rebuilt)
=================================  =================================  ==================

.. note::

   ``revectorize_database`` opens the DB using its **current** vectorizer,
   decrypts every chunk, re-embeds the same text with the new vectorizer, and
   atomically updates ``vectorizer_info`` + ``store_config``. This is the
   right tool when you want to swap models without touching chunk layout.


``revectorize_database`` — swap the model, keep chunks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   with safe_store.SafeStore.from_db_auto("knowledge.db") as store:
       store.revectorize_database(
           new_vectorizer_name="openai",
           new_vectorizer_config={"model": "text-embedding-3-small"},
       )

The ``store`` instance is mutated in place — ``store.vectorizer_name`` and
``store.vectorizer_config`` reflect the new values on success. Subsequent
queries use the new vectorizer transparently.

.. warning::

   ``revectorize_database`` uses the *current* vectorizer to read the plaintext
   back from the DB. On an **encrypted** datastore, the current
   ``encryption_key`` must be supplied — you cannot revectorize a datastore
   you don't possess the key for.

   .. code-block:: python

      # ``from_db_auto`` is a CLASSMETHOD — it takes db_path first, then any
      # SafeStore(...) kwargs (e.g. ``encryption_key``). Do NOT chain it on a
      # SafeStore instance: that constructs a throwaway store (which would
      # raise ConfigurationError on a vectorizer mismatch) before the class
      # method even runs.
      with safe_store.SafeStore.from_db_auto(
          "secure.db",
          encryption_key="my-key",
      ) as store:
          store.revectorize_database(
              new_vectorizer_name="onnx",
              new_vectorizer_config={"model": "BAAI/bge-small-en-v1.5"},
          )


``rechunk_and_vectorize`` — swap chunks AND model in one pass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   with safe_store.SafeStore.from_db_auto("knowledge.db") as store:
       store.rechunk_and_vectorize(
           new_chunk_size=2048,
           new_chunk_overlap=64,
           new_chunking_strategy="token",
           new_vectorizer_name="st",
           new_vectorizer_config={"model": "all-mpnet-base-v2"},
           vectorize_with_metadata=True,
           metadata_mode="parsed",
           batch_size=50,
       )


Safety workflow
~~~~~~~~~~~~~~~

Before any migration on a production datastore, take a byte-exact backup:

.. code-block:: python

   with safe_store.SafeStore.from_db_auto("knowledge.db") as store:
       store.export_db_file("knowledge_backup_2026-11-15.db")

``export_db_file`` uses SQLite's online-backup API to produce a single,
WAL-free ``.db`` file that is immediately usable.


Vectorizer model configuration reference
========================================

Each vectorizer backend is configured through ``vectorizer_config`` (a plain
``dict``). Below are complete worked examples for every backend that requires
a model choice. Backends marked as *model-free* (``grepper``) only need
``vectorizer_name``.

Sentence-Transformers (``st``)
------------------------------

**Local PyTorch + HuggingFace models.** First use auto-installs
``torch`` + ``sentence-transformers`` via pipmaster (30–90 s).

.. code-block:: python

   store = safe_store.SafeStore(
       db_path="knowledge.db",
       vectorizer_name="st",
       vectorizer_config={
           "model": "all-MiniLM-L6-v2",  # 384-dim, ~90 MB
       },
   )

**Common model choices** (any HuggingFace ``sentence-transformers/*`` model
works too):

=============================== ===== ==========================================
Model                            Dim   Notes
=============================== ===== ==========================================
``all-MiniLM-L6-v2``             384   Fast, English, general-purpose (default).
``all-mpnet-base-v2``            768   Higher quality, ~2× slower.
``multi-qa-mpnet-base-dot-v1``   768   Tuned for Q/A and retrieval.
``all-distilroberta-v1``         768   RoBERTa-based, better long-text handling.
``paraphrase-albert-small-v2``   768   Very small, paraphrase-tuned.
``LaBSE``                        768   Multilingual (109 languages). ~1.8 GB.
``all-roberta-large-v1``        1024   Large, highest quality, slowest.
=============================== ===== ==========================================

**Shared-server mode** for multi-process deployments — one model in RAM
instead of one per worker:

.. code-block:: python

   store = safe_store.SafeStore(
       db_path="knowledge.db",
       vectorizer_name="st",
       vectorizer_config={
           "model": "all-MiniLM-L6-v2",
           "use_shared_server": True,
           "port": 8765,
           "idle_timeout": 0.0,       # 0 = process stays alive until shutdown
       },
   )

**GPU selection:**

.. code-block:: python

   "vectorizer_config": {
       "model": "all-mpnet-base-v2",
       "device": "cuda:0",             # or "cpu", "mps"
   }

**Custom model cache location:**

.. code-block:: python

   store = safe_store.SafeStore(
       db_path="knowledge.db",
       vectorizer_name="st",
       vectorizer_config={"model": "all-MiniLM-L6-v2"},
       cache_folder=r"D:\hf_models",
   )

**Migrating a datastore to a different ST model:**

.. code-block:: python

   with safe_store.SafeStore.from_db_auto("knowledge.db") as store:
       store.revectorize_database(
           new_vectorizer_name="st",
           new_vectorizer_config={"model": "all-mpnet-base-v2"},
       )


ONNX Runtime (``onnx``)
-----------------------

**No PyTorch required** — uses ``fastembed`` (ONNX Runtime). Faster cold start
than ``st`` and much smaller install footprint. Auto-installs ``fastembed`` on
first use.

.. code-block:: python

   store = safe_store.SafeStore(
       db_path="knowledge.db",
       vectorizer_name="onnx",
       vectorizer_config={
           "model": "BAAI/bge-small-en-v1.5",  # 384-dim
       },
   )

**Common model choices** (any ``fastembed``-supported model works):

=============================== ===== ==========================================
Model                            Dim   Notes
=============================== ===== ==========================================
``BAAI/bge-small-en-v1.5``       384   Default. English, high-quality/small.
``BAAI/bge-base-en-v1.5``        768   Higher quality.
``BAAI/bge-large-en-v1.5``      1024   Highest quality, slowest.
``intfloat/multilingual-e5-small`` 384 Multilingual (100+ languages).
``sentence-transformers/all-MiniLM-L6-v2`` 384 fastembed-wrapped version of ST's default.
=============================== ===== ==========================================

**GPU acceleration** (requires ``fastembed-gpu`` instead of ``fastembed``):

.. code-block:: python

   "vectorizer_config": {
       "model": "BAAI/bge-small-en-v1.5",
       "providers": ["CUDAExecutionProvider"],
   }

**Tuning:**

.. code-block:: python

   "vectorizer_config": {
       "model": "BAAI/bge-small-en-v1.5",
       "threads": 8,                # CPU thread count
       "batch_size": 64,            # inference batch size
   }

**Migrating ST → ONNX** (a very common operation when switching off PyTorch):

.. code-block:: python

   with safe_store.SafeStore.from_db_auto("knowledge.db") as store:
       store.revectorize_database(
           new_vectorizer_name="onnx",
           new_vectorizer_config={"model": "BAAI/bge-small-en-v1.5"},
       )


OpenAI (``openai``)
-------------------

**Remote API** — no local inference. Requires the ``openai`` SDK (auto-installed
on first use) and an API key.

.. code-block:: python

   store = safe_store.SafeStore(
       db_path="knowledge.db",
       vectorizer_name="openai",
       vectorizer_config={
           "model": "text-embedding-3-small",
           # Preferred: read from OPENAI_API_KEY env var and omit `api_key`.
           # "api_key": "sk-...",
       },
   )

**Common model choices:**

=============================== ===== ==========================================
Model                            Dim   Notes
=============================== ===== ==========================================
``text-embedding-3-small``      1536   Default. Lowest cost, fast.
``text-embedding-3-large``      3072   Highest quality.
``text-embedding-ada-002``      1536   Legacy.
=============================== ===== ==========================================

**OpenAI-compatible endpoints** (Azure, Groq, Together, LiteLLM, Ollama):

.. code-block:: python

   "vectorizer_config": {
       "model": "text-embedding-3-small",
       "api_key": "sk-...",
       "base_url": "https://api.your-proxy.example.com/v1",
   }

**Migrating to OpenAI:**

.. code-block:: python

   with safe_store.SafeStore.from_db_auto("knowledge.db") as store:
       store.revectorize_database(
           new_vectorizer_name="openai",
           new_vectorizer_config={"model": "text-embedding-3-small"},
       )


Ollama (``ollama``)
-------------------

**Local Ollama daemon** — runs any embedding model served by Ollama. Auto-installs
the ``ollama`` client SDK on first use. Ollama itself must be running.

.. code-block:: python

   store = safe_store.SafeStore(
       db_path="knowledge.db",
       vectorizer_name="ollama",
       vectorizer_config={
           "model": "nomic-embed-text",
           # "host": "http://localhost:11434",   # default
       },
   )

**Common model choices** (any model already pulled in the Ollama daemon works —
run ``ollama list`` to check):

=============================== ===== ==========================================
Model                            Dim   Notes
=============================== ===== ==========================================
``nomic-embed-text``             768   Best general-purpose. Recommended.
``mxbai-embed-large``           1024   Higher quality, slower.
``snowflake-arctic-embed``      1024   Multilingual, retrieval-tuned.
``all-minilm``                   384   Ollama's port of ST's default.
=============================== ===== ==========================================

**Remote Ollama** (behind auth proxy):

.. code-block:: python

   "vectorizer_config": {
       "model": "nomic-embed-text",
       "host": "https://ollama.internal.company.com",
       "api_key": "...",          # optional, sent as Bearer
   }

**Migrating to Ollama:**

.. code-block:: python

   with safe_store.SafeStore.from_db_auto("knowledge.db") as store:
       store.revectorize_database(
           new_vectorizer_name="ollama",
           new_vectorizer_config={"model": "nomic-embed-text"},
       )


Cohere (``cohere``)
-------------------

**Remote API** — requires the ``cohere`` SDK (auto-installed) and an API key.

.. code-block:: python

   store = safe_store.SafeStore(
       db_path="knowledge.db",
       vectorizer_name="cohere",
       vectorizer_config={
           "model": "embed-english-v3.0",
           # Preferred: read from COHERE_API_KEY env var.
           # "api_key": "...",
           "input_type": "search_document",   # or "search_query"
           "truncate": "END",
       },
   )

**Common model choices:**

=============================== ===== ==========================================
Model                            Dim   Notes
=============================== ===== ==========================================
``embed-english-v3.0``          1024   Default. English.
``embed-english-light-v3.0``     384   Fast, cheap, English only.
``embed-multilingual-v3.0``     1024   Multilingual (100+ languages).
``embed-multilingual-light-v3.0`` 384 Fast multilingual.
=============================== ===== ==========================================

.. tip::

   Cohere distinguishes **document** embeddings from **query** embeddings via
   ``input_type``. Set ``"search_document"`` when the datastore is used for
   ingestion and ``"search_query"`` when the datastore serves query-side
   retrieval. If one datastore handles both, keep the default
   ``"search_document"``; Cohere's v3 models are tolerant of asymmetric usage.

**Migrating to Cohere:**

.. code-block:: python

   with safe_store.SafeStore.from_db_auto("knowledge.db") as store:
       store.revectorize_database(
           new_vectorizer_name="cohere",
           new_vectorizer_config={"model": "embed-multilingual-v3.0"},
       )


Lollms (``lollms``)
------------------

**OpenAI-compatible API** — works with Lollms-WebUI, vLLM, LiteLLM, or any
OpenAI-compatible embedding endpoint. Requires the ``openai`` SDK + optionally
``lollms-client``. Auto-installs on first use.

.. code-block:: python

   store = safe_store.SafeStore(
       db_path="knowledge.db",
       vectorizer_name="lollms",
       vectorizer_config={
           "model": "nomic-embed-text",
           "base_url": "http://localhost:9600",
           "api_key": "not_needed",     # most local Lollms servers ignore this
       },
   )

**Common endpoints:**

=============================== ==========================================
Endpoint                         Backend
=============================== ==========================================
``http://localhost:9600``        Lollms-WebUI default
``http://localhost:8000/v1``     vLLM OpenAI-compatible
``http://localhost:4000/v1``     LiteLLM proxy
=============================== ==========================================

The ``model`` string must match whatever embedding model the Lollms server
exposes. Query the server's ``/v1/models`` endpoint to discover valid names,
or use the ``openai`` SDK directly against the same ``base_url``:

.. code-block:: python

   # ``SafeStore.list_models(vectorizer_name="lollms", ...)`` returns ``[]`` —
   # the Lollms vectorizer intentionally does not advertise a static model list
   # (any embedding model the endpoint exposes is valid). Discover dynamically:
   from openai import OpenAI
   client = OpenAI(base_url="http://localhost:9600", api_key="not_needed")
   print([m.id for m in client.models.list().data])

   # For backends that DO advertise a static list, use SafeStore.list_models:
   #   safe_store.SafeStore.list_models("ollama")     -> ["nomic-embed-text", ...]
   #   safe_store.SafeStore.list_models("openai")     -> ["text-embedding-3-small", ...]

**Migrating to a Lollms endpoint:**

.. code-block:: python

   with safe_store.SafeStore.from_db_auto("knowledge.db") as store:
       store.revectorize_database(
           new_vectorizer_name="lollms",
           new_vectorizer_config={
               "model": "nomic-embed-text",
               "base_url": "http://localhost:9600",
           },
       )


TF-IDF (``tfidf`` / ``tf_idf``)
-------------------------------

**Data-dependent** — there is no pre-trained model. The vectorizer learns a
vocabulary from the chunks you ingest. Config is minimal.

.. code-block:: python

   store = safe_store.SafeStore(
       db_path="knowledge.db",
       vectorizer_name="tfidf",
       # no vectorizer_config required
   )

.. note::

   TF-IDF is fitted on the *first* document ingested and grows with each
   subsequent document. Its dimension is therefore not known until the first
   ingestion. Because the vocabulary is fixed after fitting, ``tfidf`` is
   **not recommended** for evolving datastores. Prefer ``st`` / ``onnx`` /
   ``ollama`` for production. ``tfidf`` is best used for offline baselines or
   tests.

**Migrating *from* TF-IDF** (recommended — TF-IDF vectors are not comparable
to neural embeddings):

.. code-block:: python

   with safe_store.SafeStore.from_db_auto("knowledge.db") as store:
       store.revectorize_database(
           new_vectorizer_name="st",
           new_vectorizer_config={"model": "all-MiniLM-L6-v2"},
       )


Grepper (``grepper``)
--------------------

**Model-free** — a keyword inverted index with markdown-tree extraction. It
stores placeholder 1-dim vectors for compatibility but performs real search via
its own inverted-index hooks. No model to configure, no significant setup cost.

.. code-block:: python

   store = safe_store.SafeStore(
       db_path="knowledge.db",
       vectorizer_name="grepper",
       # no vectorizer_config
   )

.. tip::

   ``grepper`` is best for **small, markdown-heavy datastores** where you want
   exact-keyword + header-path retrieval without any ML dependency. It's not a
   substitute for dense embeddings on semantic queries.

**Migrating *from* Grepper** (upgrade to real embeddings):

.. code-block:: python

   with safe_store.SafeStore.from_db_auto("knowledge.db") as store:
       store.revectorize_database(
           new_vectorizer_name="onnx",
           new_vectorizer_config={"model": "BAAI/bge-small-en-v1.5"},
       )


Reopening a datastore whose vectorizer differs
---------------------------------------------

If a datastore was indexed with ``st`` but the surrounding application defaults
to ``onnx``, the strict ``SafeStore(...)`` constructor raises:

.. code-block:: text

   ConfigurationError: Database at '...' has an incompatible vectorizer:
   'st:{"model":"all-MiniLM-L6-v2"}'. This instance is configured with
   'onnx:{"model":"BAAI/bge-small-en-v1.5"}'.

This is a **safety guard** — reusing a datastore with a different vectorizer
would silently corrupt query results. Three ways out:

**A. Open with the correct vectorizer** (read-only inspection):

.. code-block:: python

   with safe_store.SafeStore(
       "knowledge.db",
       vectorizer_name="st",
       vectorizer_config={"model": "all-MiniLM-L6-v2"},
   ) as store:
       store.info()

**B. Let SafeStore pick it** (recommended for UI / inspection tools):

.. code-block:: python

   with safe_store.SafeStore.from_db_auto("knowledge.db") as store:
       store.info()
       # The DB's own vectorizer identity is used automatically.

**C. Migrate the datastore to the new vectorizer** (permanent):

.. code-block:: python

   with safe_store.SafeStore.from_db_auto("knowledge.db") as store:
       store.revectorize_database(
           new_vectorizer_name="onnx",
           new_vectorizer_config={"model": "BAAI/bge-small-en-v1.5"},
       )
   # Future SafeStore(..., vectorizer_name="onnx", ...) opens cleanly.

Regardless of choice, restore the strict constructor once the mismatch is
resolved — ``from_db_auto`` is intended for inspection / generic-open paths,
not for ingestion pipelines.


See also
--------

- :doc:`quickstart` — create a datastore from scratch.
- :doc:`encryption` — encrypted datastores (required for DNA key scenarios).
- :doc:`logging` — control ``ASCIIColors`` verbosity.
- :doc:`shared_model_server_architecture` — multi-process deployments with ``st``.