import huggingface_hub
import warnings

from mlx.core.metal import device_info
from typing import List, Optional


class Models:
    def __init__(
        self, cache_dir: Optional[str] = None, max_vram_bytes: Optional[int] = None
    ):
        self._cache_dir = cache_dir
        self._max_vram = int(
            max_vram_bytes or device_info()["max_recommended_working_set_size"]
        )

    def _will_fit_in_vram(self, model: ...) -> bool:
        if not model.safetensors:
            return False
        safetensors_mem_req = 0
        for key in model.safetensors.parameters.keys():
            safetensors_mem_req += model.safetensors.parameters.get(key) * int(
                "".join(filter(str.isdigit, key))
            )
        safetensors_mem_req /= 2**3  # 2**3 converts to bytes
        return self._max_vram * 0.9 > safetensors_mem_req

    def add(self, model: str) -> None:
        huggingface_hub.snapshot_download(
            cache_dir=self._cache_dir,
            repo_id=model,
            allow_patterns=["*.json", "*.md", "*.txt", "*.safetensors"],
            ignore_patterns=["*.bin.index.json"],
        )

    def delete(self, model: str) -> None:
        cache = huggingface_hub.scan_cache_dir(cache_dir=self._cache_dir)
        model_revisions = []
        for repo in cache.repos:
            if repo.repo_id == model:
                model_revisions = [revision.commit_hash for revision in repo.revisions]
        delete_strategy = cache.delete_revisions(*model_revisions)
        delete_strategy.execute()

    def list(self) -> List[str]:
        return self.search_local("")

    def search_online(self, query: str, max_results: int = 5) -> List[str]:
        models = list(
            huggingface_hub.list_models(
                model_name=query,
                expand=["safetensors"],
                gated=False,
                limit=max_results,
                sort="trending_score",
                task="text-generation",
            )
        )
        return [model.id for model in filter(self._will_fit_in_vram, models)]

    def search_local(self, query: str) -> List[str]:
        models = []
        query_terms = [t.lower() for t in query.split()]
        for repo in huggingface_hub.scan_cache_dir().repos:
            matches = [t in repo.repo_id.lower() for t in query_terms]
            if all(matches):
                revisions = list(repo.revisions)
                if len(revisions) > 1:
                    warnings.warn("Multiple revisions detected.")
                revision = revisions[0]
                safetensors_mem_req = sum(
                    [
                        f.size_on_disk
                        for f in revision.files
                        if f.file_name.endswith(".safetensors")
                    ]
                )
                if self._max_vram * 0.9 > safetensors_mem_req:
                    models.append(repo.repo_id)
        return models
