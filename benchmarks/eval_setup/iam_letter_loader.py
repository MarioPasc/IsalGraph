"""IAM Letter dataset loader (GXL/CXL format).

Parses the Zenodo IAM Letter dataset into NetworkX graphs with
topology only (x,y node coordinates are stripped).

Reference:
    Riesen & Bunke (2008). IAM Graph Database Repository for Graph
    Based Pattern Recognition and Machine Learning.
    https://zenodo.org/records/13763793
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import networkx as nx

logger = logging.getLogger(__name__)

SPLITS = ("train", "validation", "test")


@dataclass
class IAMLetterDataset:
    """Parsed IAM Letter dataset for one distortion level."""

    graphs: list[nx.Graph] = field(default_factory=list)
    graph_ids: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    splits: list[str] = field(default_factory=list)
    level: str = ""


def parse_cxl(cxl_path: str) -> list[tuple[str, str]]:
    """Parse a CXL index file.

    Args:
        cxl_path: Path to the .cxl file.

    Returns:
        List of (filename, class_label) tuples.
    """
    tree = ET.parse(cxl_path)
    root = tree.getroot()
    entries: list[tuple[str, str]] = []
    for print_el in root.iter("print"):
        filename = print_el.get("file", "")
        class_label = print_el.get("class", "")
        if filename:
            entries.append((filename, class_label))
    return entries


def parse_gxl(gxl_path: str) -> nx.Graph:
    """Parse a single GXL file into a NetworkX graph (topology only).

    Node attributes (x, y coordinates) are stripped. Node IDs are
    mapped from GXL format ('_0', '_1', ...) to integers (0, 1, ...).

    Args:
        gxl_path: Path to the .gxl file.

    Returns:
        Undirected NetworkX graph with integer node labels.
    """
    tree = ET.parse(gxl_path)
    root = tree.getroot()
    graph_el = root.find(".//graph")
    if graph_el is None:
        raise ValueError(f"No <graph> element found in {gxl_path}")

    # Map GXL node IDs to integers
    nodes = graph_el.findall("node")
    id_map: dict[str, int] = {}
    for i, node in enumerate(nodes):
        gxl_id = node.get("id", f"_unknown_{i}")
        id_map[gxl_id] = i

    g = nx.Graph()
    g.add_nodes_from(range(len(nodes)))

    for edge in graph_el.findall("edge"):
        src_gxl = edge.get("from", "")
        tgt_gxl = edge.get("to", "")
        if src_gxl in id_map and tgt_gxl in id_map:
            src = id_map[src_gxl]
            tgt = id_map[tgt_gxl]
            if src != tgt:
                g.add_edge(src, tgt)
        else:
            logger.warning(
                "Edge references unknown node in %s: %s -> %s",
                gxl_path,
                src_gxl,
                tgt_gxl,
            )

    return g


def load_iam_letter(
    base_dir: str,
    level: str,
    splits: tuple[str, ...] = SPLITS,
) -> IAMLetterDataset:
    """Load IAM Letter dataset for one distortion level.

    Merges all requested splits into a single dataset. Each graph
    retains metadata about its split origin and class label.

    Args:
        base_dir: Path to the Letter directory containing LOW/MED/HIGH.
        level: Distortion level ('LOW', 'MED', or 'HIGH').
        splits: Which splits to load.

    Returns:
        IAMLetterDataset with all graphs for the level.
    """
    level_dir = os.path.join(base_dir, level)
    if not os.path.isdir(level_dir):
        raise FileNotFoundError(f"IAM Letter directory not found: {level_dir}")

    dataset = IAMLetterDataset(level=level)
    seen_ids: set[str] = set()

    for split in splits:
        cxl_path = os.path.join(level_dir, f"{split}.cxl")
        if not os.path.isfile(cxl_path):
            logger.warning("CXL file not found: %s", cxl_path)
            continue

        entries = parse_cxl(cxl_path)
        logger.info("Loading %s/%s: %d entries from %s", level, split, len(entries), cxl_path)

        for filename, class_label in entries:
            # Avoid duplicates across splits
            if filename in seen_ids:
                continue
            seen_ids.add(filename)

            gxl_path = os.path.join(level_dir, filename)
            if not os.path.isfile(gxl_path):
                logger.warning("GXL file not found: %s", gxl_path)
                continue

            try:
                g = parse_gxl(gxl_path)
            except Exception:
                logger.exception("Failed to parse %s", gxl_path)
                continue

            # Use filename without extension as graph ID
            graph_id = os.path.splitext(filename)[0]
            dataset.graphs.append(g)
            dataset.graph_ids.append(graph_id)
            dataset.labels.append(class_label)
            dataset.splits.append(split)

    logger.info(
        "Loaded IAM Letter %s: %d graphs across %s",
        level,
        len(dataset.graphs),
        ", ".join(splits),
    )
    return dataset
