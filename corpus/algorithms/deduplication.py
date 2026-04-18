import hashlib
from collections import defaultdict
from typing import List


def exact_dedup(raw_dataset: List[str], num_parallel_agents=4):
    """
    Prototype for parallel exact deduplication.
    Input: documents is a list of strings.
    Output: a list of indices representing the deduplicated dataset
    """

    hashed = map(lambda x: hashlib.md5(x.encode()).digest(), raw_dataset)
    hashed = enumerate(hashed)

    buckets = defaultdict(list)
    for text_id, hash_value in hashed:
        # divide among num_parallel_agents
        rank = int.from_bytes(hash_value, "big") % num_parallel_agents

        buckets[rank].append((text_id, hash_value))

    # in parallel
    indices = set()
    for subset in buckets.values():
        seen = set()
        for text_id, hash_value in subset:
            if hash_value not in seen:
                indices.add(text_id)
                seen.add(hash_value)

    return sorted(indices)
