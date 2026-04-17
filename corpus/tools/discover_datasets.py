from huggingface_hub import HfApi


def find_dataset():
    api = HfApi()
    res = api.list_datasets(sort="trending_score", limit=100)
    for ds in res:
        print(f"{ds.id} - Downloads (last 30d): {ds.downloads}")
