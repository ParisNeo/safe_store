from typing import List, Dict, Any, Optional
import math
import numpy as np

def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    weights: Optional[List[float]] = None,
    k: int = 60,
    top_k: int = 5,
    min_relevance_percent: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Score-Calibrated Multi-Modal Reciprocal Rank Fusion.
    
    Combines dense semantic similarity, sparse lexical BM25, and graph retrieval
    while preserving true modality score magnitudes calibrated on a [0.0, 100.0] scale.
    """
    if not ranked_lists or not any(ranked_lists):
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)
    elif len(weights) != len(ranked_lists):
        raise ValueError("Length of weights must match length of ranked_lists.")

    total_configured_weight = sum(weights) if sum(weights) > 0 else 1.0

    # Key -> Aggregated Data
    candidates: Dict[Any, Dict[str, Any]] = {}

    for list_idx, (ranked_list, weight) in enumerate(zip(ranked_lists, weights)):
        if not ranked_list:
            continue

        for rank, item in enumerate(ranked_list, start=1):
            raw_key = item.get("chunk_id") if "chunk_id" in item else item.get("file_path", id(item))
            key = int(raw_key) if isinstance(raw_key, (int, np.integer)) or (isinstance(raw_key, str) and raw_key.isdigit()) else raw_key

            
            # Extract underlying modality score (0-100 grade)
            modality_score = float(
                item.get("relevance_score", 
                item.get("similarity_percent", 
                item.get("similarity_score", 
                item.get("score", 0.0))))
            )
            # If similarity_score was raw cosine [-1, 1], normalize to [0, 100]
            if -1.0 <= modality_score <= 1.0 and "relevance_score" not in item and "similarity_percent" not in item:
                modality_score = ((modality_score + 1.0) / 2.0) * 100.0

            modality_score = max(0.0, min(100.0, modality_score))

            # Positional dampening factor
            rank_factor = float(k + 1) / float(k + rank)
            raw_rrf_increment = float(weight) / float(k + rank)

            if key not in candidates:
                candidates[key] = {
                    "payload": dict(item),
                    "raw_rrf": 0.0,
                    "weighted_score_sum": 0.0,
                    "matched_weight_sum": 0.0,
                    "modalities_hit": 0
                }

            entry = candidates[key]
            entry["raw_rrf"] += raw_rrf_increment
            entry["weighted_score_sum"] += (weight * modality_score * rank_factor)
            entry["matched_weight_sum"] += weight
            entry["modalities_hit"] += 1

    # Score calibration
    fused_results = []
    for key, data in candidates.items():
        matched_w = data["matched_weight_sum"]
        if matched_w <= 0:
            continue

        # Modality score invariant: take the maximum dampened modality score or weighted average
        base_grade = data["weighted_score_sum"] / matched_w

        # Multi-modal agreement bonus: matching across multiple channels increases confidence
        agreement_bonus = 5.0 * max(0, data["modalities_hit"] - 1)

        # Final score retains full strength of the best modality, boosted by cross-modal agreement
        final_relevance = max(0.0, min(100.0, base_grade + agreement_bonus))
        final_relevance = round(final_relevance, 2)

        if final_relevance < min_relevance_percent:
            continue

        payload = data["payload"]
        payload["raw_rrf_score"] = float(round(data["raw_rrf"], 6))
        payload["fused_score"] = float(round(data["raw_rrf"], 6))
        payload["relevance_score"] = float(final_relevance)
        payload["similarity_percent"] = float(final_relevance)
        
        fused_results.append(payload)

    # Sort descending by calibrated relevance score
    fused_results.sort(key=lambda x: x["relevance_score"], reverse=True)

    return fused_results[:top_k] if top_k > 0 else fused_results


def weighted_score_fusion(
    scored_lists: List[List[Dict[str, Any]]],
    weights: Optional[List[float]] = None,
    top_k: int = 5,
    min_relevance_percent: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Score-Calibrated Weighted Linear Combination across multiple retrieval channels.
    """
    if not scored_lists or not any(scored_lists):
        return []

    if weights is None:
        weights = [1.0 / len(scored_lists)] * len(scored_lists)
    elif len(weights) != len(scored_lists):
        raise ValueError("Length of weights must match length of scored_lists.")

    total_w = sum(weights)
    norm_weights = [w / total_w for w in weights] if total_w > 0 else weights

    candidates: Dict[Any, Dict[str, Any]] = {}

    for list_idx, (scored_list, w) in enumerate(zip(scored_lists, norm_weights)):
        if not scored_list:
            continue

        for item in scored_list:
            key = item.get("chunk_id") if "chunk_id" in item else item.get("file_path", id(item))
            score_val = float(
                item.get("relevance_score", 
                item.get("similarity_percent", 
                item.get("similarity_score", 
                item.get("score", 0.0))))
            )
            if -1.0 <= score_val <= 1.0 and "relevance_score" not in item and "similarity_percent" not in item:
                score_val = ((score_val + 1.0) / 2.0) * 100.0
            
            score_val = max(0.0, min(100.0, score_val))

            if key not in candidates:
                candidates[key] = {
                    "payload": dict(item),
                    "accumulated_score": 0.0
                }
            candidates[key]["accumulated_score"] += (score_val * w)

    results = []
    for key, data in candidates.items():
        grade = round(max(0.0, min(100.0, data["accumulated_score"])), 2)
        if grade < min_relevance_percent:
            continue

        payload = data["payload"]
        payload["relevance_score"] = float(grade)
        payload["similarity_percent"] = float(grade)
        payload["fused_score"] = float(grade)
        results.append(payload)

    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results[:top_k] if top_k > 0 else results