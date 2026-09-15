"""Protocols and result types for agent tools."""

from dataclasses import dataclass
from typing import Any, Dict, Protocol


@dataclass(frozen=True)
class SegmentationResult:
    """Serialized output produced by a segmentation tool."""

    json_path: str
    image_path: str
    data: Dict[str, Any]


class SegmentationTool(Protocol):
    """Tool used by the agent to ground a text phrase in an image."""

    def segment_phrase(
        self,
        image_path: str,
        text_prompt: str,
        output_dir: str,
        verbose: bool = False,
    ) -> SegmentationResult:
        """Return masks, boxes, scores, and a rendered visualization."""
