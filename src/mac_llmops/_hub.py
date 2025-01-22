import huggingface_hub
import warnings

from mlx.core.metal import device_info
from typing import List, Optional


class Models:
    """Manage local model caches.

    Args:
        cache_dir: optionally change the path to the cache directory from the HuggingFace default.
        max_vram_bytes: optionally override the maximum GPU memory available for running LLMs. The default value is derived from the local GPU.
    """

    def __init__(
        self, cache_dir: Optional[str] = None, max_vram_bytes: Optional[int] = None
    ):
        self._cache_dir = cache_dir
        self._max_vram = int(
            max_vram_bytes or device_info()["max_recommended_working_set_size"]
        )

    def _will_fit_in_vram(self, model: ...) -> bool:
        # does not take kv cache requirements into account
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
        """Add a model from a HuggingFace hub to your local model cache.

        Args:
            model: name of model to add.
        """
        huggingface_hub.snapshot_download(
            cache_dir=self._cache_dir,
            repo_id=model,
            allow_patterns=["*.json", "*.md", "*.txt", "*.safetensors"],
            ignore_patterns=["*.bin.index.json"],
        )

    def delete(self, model: str) -> None:
        """Delete a model from your local model cache.

        Args:
            model: name of model to delete.
        """
        cache = huggingface_hub.scan_cache_dir(cache_dir=self._cache_dir)
        model_revisions = []
        for repo in cache.repos:
            if repo.repo_id == model:
                model_revisions = [revision.commit_hash for revision in repo.revisions]
        delete_strategy = cache.delete_revisions(*model_revisions)
        delete_strategy.execute()

    def list(self) -> List[str]:
        """Show all models in your local cache that will fit in GPU memory."""
        return self.search_local("")

    def search_local(self, query: str) -> List[str]:
        """Search your local model cache for models that will fit in GPU memory.

        Args:
            query: space separated search terms to match in the model name.

        Returns:
            A list of models that contain all search terms in their name.
        """
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
                    models.append({"model": repo.repo_id, "size_gb": round(repo.size_on_disk / 2**30, 2)})
        return models

    def search_online(self, query: str, max_results: int = 20) -> List[str]:
        """Search a HuggingFace hub for models that will fit in GPU memory.

        Args:
            query: space separated search terms to match in the model name.
            max_results: number of results returned from the HuggingFace hub, before filtering for GPU memory requirements.

        Returns:
            A list of models that contain all search terms in their name.
        """
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
