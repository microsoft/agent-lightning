# Copyright (c) Microsoft. All rights reserved.

"""Runner-readiness publication and lookup routes."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException

from agentlightning.k8s import normalize_image_reference
from agentlightning.schemas import K8sImageReadinessReport, K8sImageReadinessSnapshot
from agentlightning.server.store import _runner_readiness

router = APIRouter(tags=["runner-readiness"])
_K8S_KEY = "k8s"


@router.put("/runner-readiness/k8s", response_model=K8sImageReadinessSnapshot)
async def publish_k8s_image_readiness(body: K8sImageReadinessReport) -> K8sImageReadinessSnapshot:
    """Publish a leased image inventory using the server's clock."""
    now = time.time()
    try:
        images = sorted({normalize_image_reference(image) for image in body.images})
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    snapshot = K8sImageReadinessSnapshot(
        images=images,
        node_count=body.node_count,
        observed_at=now,
        expires_at=now + body.lease_seconds,
    )
    _runner_readiness[_K8S_KEY] = snapshot
    return snapshot


@router.get("/runner-readiness/k8s", response_model=K8sImageReadinessSnapshot)
async def get_k8s_image_readiness() -> K8sImageReadinessSnapshot:
    """Return the current snapshot, failing closed when absent or expired."""
    snapshot = _runner_readiness.get(_K8S_KEY)
    if snapshot is None:
        raise HTTPException(status_code=503, detail="K8s image readiness has not been published")
    if snapshot.expires_at <= time.time():
        raise HTTPException(status_code=503, detail="K8s image readiness snapshot has expired")
    return snapshot
