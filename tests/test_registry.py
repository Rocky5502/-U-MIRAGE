from umirage.datasets import REGISTRY

def test_registry_ids_unique():
    assert len(REGISTRY)==len(set(REGISTRY))
    assert {"mioh","vischainbench","mihbench"}.issubset(REGISTRY)
