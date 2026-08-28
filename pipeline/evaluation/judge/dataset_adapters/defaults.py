"""Construction of the built-in raw dataset adapter registry."""

from pipeline.evaluation.judge.dataset_adapters.hotpotqa import HotpotQAAdapter
from pipeline.evaluation.judge.dataset_adapters.musique import MusiqueAdapter
from pipeline.evaluation.judge.dataset_adapters.narrativeqa import NarrativeQAAdapter
from pipeline.evaluation.judge.dataset_adapters.registry import DatasetAdapterRegistry
from pipeline.evaluation.judge.dataset_adapters.sample import SampleAdapter
from pipeline.evaluation.judge.dataset_adapters.two_wiki import TwoWikiAdapter


def build_default_dataset_registry() -> DatasetAdapterRegistry:
    registry = DatasetAdapterRegistry()
    registry.register(HotpotQAAdapter())
    registry.register(TwoWikiAdapter())
    registry.register(MusiqueAdapter())
    registry.register(NarrativeQAAdapter())
    registry.register(SampleAdapter())
    return registry
