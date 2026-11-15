title: ONNX Vectorizer (fastembed)
author: ParisNeo
creation_date: 2026-11-14
last_update_date: 2026-11-14
class_name: ONNXVectorizer
description: >
  A lightweight local vectorizer running on ONNX Runtime through fastembed.
  No PyTorch, no transformers dependency: small install, fast CPU inference,
  optional GPU via execution providers. Inference only (no fine-tuning).
  Late chunking is not supported.
input_parameters:
  - name: model
    type: str
    description: "fastembed-supported model name (e.g. BAAI/bge-small-en-v1.5, intfloat/multilingual-e5-small)."
    mandatory: true
    default: "BAAI/bge-small-en-v1.5"
  - name: threads
    type: int
    description: "ONNX Runtime thread count. Unset = library default."
    mandatory: false
    default: null
  - name: providers
    type: list
    description: "ONNX execution providers, e.g. ['CUDAExecutionProvider'] (requires fastembed-gpu)."
    mandatory: false
    default: null
  - name: batch_size
    type: int
    description: "Batch size for inference."
    mandatory: false
    default: 64
  - name: reuse_model_in_process
    type: bool
    description: "Share one loaded model between instances in the same process."
    mandatory: false
    default: true