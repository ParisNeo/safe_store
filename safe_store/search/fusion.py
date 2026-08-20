from typing import List, Dict, Any, Optional
import numpy as np


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    weights: Optional[List[float]] = None,
    k: int = 60,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Combines multiple ranked lists using Reciprocal Rank Fusion (RRF).
    
    Formula: RRF_Score(d) = sum_{m in Models} ( w_m / (k + rank_m(d)) )
    where rank_m(d) is 1-based rank position in list m.
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)
    elif len(weights) != len(ranked_lists):
        raise ValueError("Length of weights must match length of ranked_lists.")

    # Aggregate scores keyed by chunk_id or file_path identifier
    fused_scores: Dict[Any, float] = {}
    item_payloads: Dict[Any, Dict[str, Any]] = {}

    for list_idx, (ranked_list, weight) in enumerate(zip(ranked_lists, weights)):
        for rank, item in enumerate(ranked_list, start=1):
            key = item.get("chunk_id") if "chunk_id" in item else item.get("file_path", id(item))
            
            rrf_val = float(weight) / float(k + rank)
            fused_scores[key] = fused_scores.get(key, 0.0) + rrf_val
            
            if key not in item_payloads:
                item_payloads[key] = dict(item)

    # Sort descending by fused score
    sorted_keys = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

    results = []
    for key in sorted_keys[:top_k] if top_k > 0 else sorted_keys:
        payload = dict(item_payloads[key])
        payload["fused_score"] = float(fused_scores[key])
        results.append(payload)

    return results


def weighted_score_fusion(
    scored_lists: List[List[Dict[str, Any]]],
    weights: Optional[List[float]] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Combines multiple scored lists using normalized min-max weighted scoring.
    """
    if not scored_lists:
        return []

    if weights is None:
        weights = [1.0 / len(scored_lists)] * len(scored_lists)
    elif len(weights) != len(scored_lists):
        raise ValueError("Length of weights must match length of scored_lists.")

    # Normalize weights
    total_w = sum(weights)
    norm_weights = [w / total_w for w in weights] if total_w > 0 else weights

    fused_scores: Dict[Any, float] = {}
    item_payloads: Dict[Any, Dict[str, Any]] = {}

    for list_idx, (scored_list, w) in enumerate(zip(scored_lists, norm_weights)):
        if not scored_list:
            continue

        raw_scores = [float(item.get("score", item.get("similarity_score", 0.0))) for item in scored_list]
        min_s = min(raw_scores)
        max_s = max(raw_scores)
        range_s = max_s - min_s if max_s > min_s else 1.0

        for item, raw_s in zip(scored_list, raw_scores):
            key = item.get("chunk_id") if "chunk_id" in item else item.get("file_path", id(item))
            norm_s = (raw_s - min_s) / range_s if max_s > min_s else 1.0
            
            fused_scores[key] = fused_scores.get(key, 0.0) + (norm_s * w)
            if key not in item_payloads:
                item_payloads[key] = dict(item)

    sorted_keys = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

    results = []
    for key in sorted_keys[:top_k] if top_k > 0 else sorted_keys:
        payload = dict(item_payloads[key])
        payload["fused_score"] = float(fused_scores[key])
        results.append(payload)

    return results