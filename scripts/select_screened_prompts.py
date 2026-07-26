"""Select an exact, balanced subset from a dense-screening manifest.

The input may contain rejected prompts and more accepted prompts than needed.
Selection first balances quotas across requested-length/evidence-depth strata,
then round-robins source cases inside each stratum. The resulting manifest is
accepted-only and can be passed directly to
``extract_teacher_targets.py --verified-manifest``.
"""
import argparse
from collections import deque
import json
import os
import random
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.screen_natural_prompts import (
    MANIFEST_KIND,
    atomic_save_manifest,
    summarize_candidates,
)


def accepted_records(manifest):
    records = [
        record for record in manifest.get("candidates", [])
        if record.get("status") == "completed" and record.get("accepted")
    ]
    identifiers = [record["candidate_id"] for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("screening manifest contains duplicate candidate IDs")
    return records


def balanced_quotas(capacities, count):
    """Allocate an exact count as evenly as capacities allow."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if count > sum(capacities.values()):
        raise ValueError(
            f"requested {count} prompts but only "
            f"{sum(capacities.values())} are available")
    quotas = {key: 0 for key in capacities}
    remaining = count
    while remaining:
        eligible = [
            key for key in capacities if quotas[key] < capacities[key]
        ]
        eligible.sort(key=lambda key: (quotas[key], key))
        for key in eligible:
            if remaining == 0:
                break
            quotas[key] += 1
            remaining -= 1
    return quotas


def _round_robin_cases(records, count, rng):
    by_case = {}
    for record in records:
        by_case.setdefault(record["source_case_id"], []).append(record)
    case_ids = sorted(by_case)
    rng.shuffle(case_ids)
    for case_id in case_ids:
        by_case[case_id].sort(
            key=lambda record: (
                int(record.get("variant", 0)),
                int(record.get("seed", 0)),
                record["candidate_id"],
            )
        )

    selected = []
    while len(selected) < count:
        progressed = False
        for case_id in case_ids:
            if by_case[case_id] and len(selected) < count:
                selected.append(by_case[case_id].pop(0))
                progressed = True
        if not progressed:
            raise ValueError("case round-robin exhausted before quota")
    return selected


def _capped_group_allocations(records, quotas, max_per_case, rng):
    """Allocate group quotas without exceeding a global semantic-case cap."""
    by_case_group = {}
    for record in records:
        key = (
            int(record["requested_length"]),
            float(record["depth"]),
        )
        by_case_group.setdefault(
            (record["source_case_id"], key), []).append(record)

    source = ("source",)
    sink = ("sink",)
    residual = {}
    adjacency = {}

    def add_edge(start, end, capacity):
        residual[(start, end)] = capacity
        residual[(end, start)] = 0
        adjacency.setdefault(start, []).append(end)
        adjacency.setdefault(end, []).append(start)

    case_ids = sorted({case_id for case_id, _ in by_case_group})
    rng.shuffle(case_ids)
    group_keys = sorted(quotas)
    for case_id in case_ids:
        case_node = ("case", case_id)
        add_edge(source, case_node, max_per_case)
        for key in group_keys:
            capacity = len(by_case_group.get((case_id, key), ()))
            if capacity:
                add_edge(case_node, ("group", key), capacity)
    for key in group_keys:
        add_edge(("group", key), sink, quotas[key])

    flow = 0
    target = sum(quotas.values())
    while flow < target:
        parents = {source: None}
        queue = deque([source])
        while queue and sink not in parents:
            node = queue.popleft()
            for neighbor in adjacency.get(node, ()):
                if neighbor not in parents and residual[(node, neighbor)] > 0:
                    parents[neighbor] = node
                    queue.append(neighbor)
        if sink not in parents:
            raise ValueError(
                f"cannot select {target} prompts with --max-per-case "
                f"{max_per_case}; increase the cap or screen more distinct "
                "semantic cases")
        amount = target - flow
        node = sink
        while parents[node] is not None:
            parent = parents[node]
            amount = min(amount, residual[(parent, node)])
            node = parent
        node = sink
        while parents[node] is not None:
            parent = parents[node]
            residual[(parent, node)] -= amount
            residual[(node, parent)] += amount
            node = parent
        flow += amount

    allocations = {}
    for case_id in case_ids:
        case_node = ("case", case_id)
        for key in group_keys:
            group_node = ("group", key)
            capacity = len(by_case_group.get((case_id, key), ()))
            if capacity:
                used = capacity - residual[(case_node, group_node)]
                if used:
                    allocations[(case_id, key)] = used
    return allocations, by_case_group


def select_balanced(records, count, seed, max_per_case=None):
    groups = {}
    for record in records:
        key = (
            int(record["requested_length"]),
            float(record["depth"]),
        )
        groups.setdefault(key, []).append(record)
    quotas = balanced_quotas(
        {key: len(rows) for key, rows in groups.items()}, count)
    rng = random.Random(seed)
    if max_per_case is not None:
        if max_per_case < 1:
            raise ValueError("max_per_case must be >= 1")
        allocations, by_case_group = _capped_group_allocations(
            records, quotas, max_per_case, rng)
        selected = []
        for (case_id, key), amount in sorted(
                allocations.items(),
                key=lambda item: (item[0][1], item[0][0])):
            choices = sorted(
                by_case_group[(case_id, key)],
                key=lambda record: (
                    int(record.get("variant", 0)),
                    int(record.get("seed", 0)),
                    record["candidate_id"],
                ),
            )
            selected.extend(choices[:amount])
        return sorted(
            selected, key=lambda record: record["candidate_id"]), quotas

    selected = []
    for key in sorted(groups):
        selected.extend(
            _round_robin_cases(groups[key], quotas[key], rng))
    return sorted(selected, key=lambda record: record["candidate_id"]), quotas


def selection_summary(records):
    summary = summarize_candidates(records)
    by_depth = {}
    by_case = {}
    by_length_depth = {}
    for record in records:
        depth = str(record["depth"])
        length = str(record["requested_length"])
        case_id = record["source_case_id"]
        by_depth[depth] = by_depth.get(depth, 0) + 1
        by_case[case_id] = by_case.get(case_id, 0) + 1
        key = f"{length}:{depth}"
        by_length_depth[key] = by_length_depth.get(key, 0) + 1
    summary.update({
        "by_depth": dict(sorted(by_depth.items(), key=lambda item: float(item[0]))),
        "by_case": dict(sorted(by_case.items())),
        "by_length_depth": dict(sorted(
            by_length_depth.items(),
            key=lambda item: (
                int(item[0].split(":")[0]),
                float(item[0].split(":")[1]),
            ),
        )),
    })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--max-per-case", type=int,
        help="global cap on selected variants from one semantic source case")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.count < 1:
        raise SystemExit("--count must be >= 1")
    with open(args.manifest, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("kind") != MANIFEST_KIND:
        raise SystemExit(
            f"{args.manifest} is not a {MANIFEST_KIND} manifest")
    try:
        accepted = accepted_records(manifest)
        selected, quotas = select_balanced(
            accepted, args.count, args.seed, args.max_per_case)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    output = {
        **manifest,
        "output": os.path.abspath(args.out),
        "source_manifest": os.path.abspath(args.manifest),
        "accepted_only": True,
        "selected_only": True,
        "selection": {
            "count": args.count,
            "seed": args.seed,
            "strategy": "balanced_length_depth_then_case_round_robin",
            "max_per_case": args.max_per_case,
            "available_accepted": len(accepted),
            "quotas": {
                f"{length}:{depth}": quota
                for (length, depth), quota in sorted(quotas.items())
            },
        },
        "candidates": selected,
        "summary": selection_summary(selected),
    }
    atomic_save_manifest(args.out, output)
    print(json.dumps(output["summary"], indent=2))
    print(f"selected manifest -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
