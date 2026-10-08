from typing import Any

import torch


def _iter_model_wrappers(model):
    current = model
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = getattr(current, "module", None)


def _get_router(model, router_name: str | None = None) -> Any:
    names = []
    if router_name:
        names.append(router_name)
    names.extend(["router", "router_dino", "router_clip"])
    for current in _iter_model_wrappers(model):
        for name in names:
            router = getattr(current, name, None)
            if router is not None:
                return router
    return None


def _iter_routers(model):
    yielded = set()
    for current in _iter_model_wrappers(model):
        for name in ("router", "router_dino", "router_clip"):
            router = getattr(current, name, None)
            if router is not None and id(router) not in yielded:
                yielded.add(id(router))
                yield router


def _latest_payload(router, router_name: str | None = None):
    if router_name and "dino" in router_name:
        return getattr(router, "_latest_routing_decision_dino", None)
    if router_name and "clip" in router_name:
        return getattr(router, "_latest_routing_decision_clip", None)

    payload = getattr(router, "_latest_routing_decision", None)
    if payload:
        return payload
    payload = getattr(router, "_latest_routing_decision_dino", None)
    if payload:
        return payload
    return getattr(router, "_latest_routing_decision_clip", None)


def clear_router_selection_metadata(model) -> None:
    for router in _iter_routers(model):
        if hasattr(router, "clear_latest_routing_decision"):
            router.clear_latest_routing_decision()
        else:
            router._latest_routing_decision = None


def _legacy_clear_router_selection_metadata(model) -> None:
    current = model
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        router = getattr(current, "router", None)
        if router is not None:
            if hasattr(router, "clear_latest_routing_decision"):
                router.clear_latest_routing_decision()
            else:
                router._latest_routing_decision = None
        current = getattr(current, "module", None)


def _count_layers(selected_layers: torch.Tensor, num_layers_available: int | None) -> dict[str, int]:
    flat_layers = selected_layers.to(torch.int64).reshape(-1)
    if flat_layers.numel() == 0:
        return {}
    if num_layers_available is None:
        num_layers_available = int(flat_layers.max().item()) + 1
    counts = torch.bincount(flat_layers, minlength=num_layers_available)
    return {
        str(layer_idx): int(count.item())
        for layer_idx, count in enumerate(counts)
        if int(count.item()) > 0
    }


def _layer_prob_mass(
    selected_layers: torch.Tensor,
    selected_probs: torch.Tensor,
    num_layers_available: int | None,
) -> dict[str, float]:
    flat_layers = selected_layers.to(torch.int64).reshape(-1)
    flat_probs = selected_probs.to(torch.float32).reshape(-1)
    if flat_layers.numel() == 0:
        return {}
    if num_layers_available is None:
        num_layers_available = int(flat_layers.max().item()) + 1
    mass = torch.zeros(num_layers_available, dtype=torch.float32)
    mass.index_add_(0, flat_layers, flat_probs)
    return {
        str(layer_idx): float(value.item())
        for layer_idx, value in enumerate(mass)
        if float(value.item()) > 0.0
    }


def _layer_frequency(layer_counts: dict[str, int]) -> dict[str, float]:
    total = sum(layer_counts.values())
    if total == 0:
        return {}
    return {
        layer_idx: count / total
        for layer_idx, count in layer_counts.items()
    }


def get_router_selection_metadata(
    model,
    batch_index: int = 0,
    router_name: str | None = None,
) -> dict[str, Any]:
    router = _get_router(model, router_name=router_name)
    if router is None:
        return {}

    payload = _latest_payload(router, router_name=router_name)
    if not payload:
        return {}

    layers = payload.get("layers")
    probs = payload.get("probs")
    if layers is None or probs is None or batch_index >= layers.shape[0]:
        return {}

    sample_layers = layers[batch_index].to(torch.int64)
    sample_probs = probs[batch_index].to(torch.float32)
    num_layers_available = payload.get("num_layers_available")

    layer_counts = _count_layers(sample_layers, num_layers_available)
    metadata = {
        "selection_mode": payload.get("selection_mode", "image_level"),
        "num_layers_available": num_layers_available,
        "num_layers_to_choose": payload.get("num_layers_to_choose"),
        "selected_layers_shape": list(sample_layers.shape),
        "selected_layers": sample_layers.tolist(),
        "selected_layer_counts": layer_counts,
        "selected_layer_frequency": _layer_frequency(layer_counts),
        "selected_layer_prob_mass": _layer_prob_mass(
            sample_layers,
            sample_probs,
            num_layers_available,
        ),
    }

    if sample_layers.ndim == 1:
        metadata["selected_layer_probs"] = [float(value) for value in sample_probs.tolist()]
    elif sample_layers.ndim >= 2:
        metadata["num_routed_patches"] = int(sample_layers.shape[0])

    reserve_layer_index = payload.get("reserve_layer_index")
    if reserve_layer_index is not None:
        metadata["reserve_layer_index"] = int(reserve_layer_index)

    hybrid_reserve_mode = payload.get("hybrid_reserve_mode")
    if hybrid_reserve_mode is not None:
        metadata["hybrid_reserve_mode"] = hybrid_reserve_mode

    hybrid_disagreement_metric = payload.get("hybrid_disagreement_metric")
    if hybrid_disagreement_metric is not None:
        metadata["hybrid_disagreement_metric"] = hybrid_disagreement_metric

    reserve_layer_prob = payload.get("reserve_layer_prob")
    if reserve_layer_prob is not None and batch_index < reserve_layer_prob.shape[0]:
        sample_reserve = reserve_layer_prob[batch_index].to(torch.float32)
        if sample_reserve.ndim == 0:
            metadata["reserve_layer_prob"] = float(sample_reserve.item())
        else:
            metadata["reserve_layer_prob_summary"] = {
                "shape": list(sample_reserve.shape),
                "mean": float(sample_reserve.mean().item()),
                "min": float(sample_reserve.min().item()),
                "max": float(sample_reserve.max().item()),
            }

    hybrid_disagreement = payload.get("hybrid_disagreement")
    if hybrid_disagreement is not None and batch_index < hybrid_disagreement.shape[0]:
        sample_disagreement = hybrid_disagreement[batch_index].to(torch.float32)
        if sample_disagreement.ndim == 0:
            metadata["hybrid_disagreement"] = float(sample_disagreement.item())
        else:
            metadata["hybrid_disagreement_summary"] = {
                "shape": list(sample_disagreement.shape),
                "mean": float(sample_disagreement.mean().item()),
                "min": float(sample_disagreement.min().item()),
                "max": float(sample_disagreement.max().item()),
            }

    return {"router": metadata}
