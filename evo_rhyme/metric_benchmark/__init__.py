"""Metric validation pipeline: schemas, protocol, segmentation, evaluation helpers."""

from evo_rhyme.metric_benchmark.schema import (
    LABEL_KEYS,
    PairwiseRecord,
    VerseRecord,
    parse_pairwise_line,
    parse_verse_line,
    validate_pairwise_record,
    validate_verse_record,
)
from evo_rhyme.metric_benchmark.protocol import (
    default_protocol_manifest,
    load_protocol_manifest,
    save_protocol_manifest,
)
from evo_rhyme.metric_benchmark.split import split_three_way

__all__ = [
    "LABEL_KEYS",
    "PairwiseRecord",
    "VerseRecord",
    "parse_pairwise_line",
    "parse_verse_line",
    "validate_pairwise_record",
    "validate_verse_record",
    "default_protocol_manifest",
    "load_protocol_manifest",
    "save_protocol_manifest",
    "split_three_way",
]
