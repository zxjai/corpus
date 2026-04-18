"""
Dolma V7 Dataset Metadata
    https://olmo-data.org/dolma-v1_7/books/books-0000.json.gz
    https://olmo-data.org/dolma-v1_7/<folder>/<file_prefix>-<NNNN>.json.gz
"""

DOLMA_BASE_URL = "https://olmo-data.org/dolma-v1_7/"

DOLMA_METADATA = {
    "Books": {
        "folder": "books",
        "file_prefix": "books",
        "num_shards": 3,
    },
    "C4 (filtered)": {
        "folder": "c4-filtered",
        "file_prefix": "c4",
        "num_shards": 171,
    },
    "Common Crawl EN (head)": {
        "folder": "cc_en_head",
        "file_prefix": "cc_en_head",
        "num_shards": 275,
    },
    "Common Crawl EN (middle)": {
        "folder": "cc_en_middle",
        "file_prefix": "cc_en_middle",
        "num_shards": 380,
    },
    "Common Crawl EN (tail)": {
        "folder": "cc_en_tail",
        "file_prefix": "cc_en_tail",
        "num_shards": 445,
    },
    "CC News (head)": {
        "folder": "cc_news_head",
        "file_prefix": "cc_news",
        "num_shards": 5,
    },
    "CC News (middle)": {
        "folder": "cc_news_middle",
        "file_prefix": "cc_news",
        "num_shards": 3,
    },
    "CC News (tail)": {
        "folder": "cc_news_tail",
        "file_prefix": "cc_news",
        "num_shards": 1,
    },
    "Falcon RefinedWeb": {
        "folder": "falcon-refinedweb-filtered",
        "file_prefix": "falcon",
        "num_shards": 500,
    },
    "AllenAI pes2o papers": {
        "folder": "pes2o",
        "file_prefix": "pes2o",
        "num_shards": 26,
    },
    "ProofPile 2 algebraic stack": {
        "folder": "proof_pile_2-algebraic_stack",
        "file_prefix": "algebraic-stack-train",
        "num_shards": 16,
    },
    "ProofPile 2 open web math": {
        "folder": "proof_pile_2-open_web_math",
        "file_prefix": "open-web-math-train",
        "num_shards": 13,
    },
    "Reddit": {
        "folder": "reddit",
        "file_prefix": "reddit",
        "num_shards": 78,
    },
    "RedPajama arXiv": {
        "folder": "redpajama-arxiv",
        "file_prefix": "arxiv",
        "num_shards": 100,
    },
    "RedPajama StackExchange": {
        "folder": "redpajama-stackexchange",
        "file_prefix": "stackexchange",
        "num_shards": 26,
    },
    "StarCoder": {
        "folder": "starcoder",
        "file_prefix": "starcoder",
        "num_shards": 49,
    },
    "Tulu FLAN": {
        "folder": "tulu_flan",
        "file_prefix": "tulu_flan",
        "num_shards": 66,
    },
    "Wikipedia": {
        "folder": "wiki",
        "file_prefix": "wiki",
        "num_shards": 2,
    },
    "WikiRef MegaWiki": {
        "folder": "wikiref_megawika",
        "file_prefix": "megawika",
        "num_shards": 262,
    },
}
