import bittensor as bt
import asyncio
from typing import Optional
from constants import CompetitionParameters, COMPETITION_SCHEDULE, ORIGINAL_COMPETITION_ID
from model.data import ModelMetadata
from model.model_tracker import ModelTracker
from model.storage.local_model_store import LocalModelStore
from model.storage.model_metadata_store import ModelMetadataStore
from model.storage.remote_model_store import RemoteModelStore
from model.utils import get_hash_of_two_strings


class ModelUpdater:
    """Checks if the currently tracked model for a hotkey matches what the miner committed to the chain."""

    def __init__(
        self,
        metadata_store: ModelMetadataStore,
        remote_store: RemoteModelStore,
        local_store: LocalModelStore,
        model_tracker: ModelTracker,
    ):
        self.metadata_store = metadata_store
        self.remote_store = remote_store
        self.local_store = local_store
        self.model_tracker = model_tracker
        self.min_block: Optional[int] = None

    def set_min_block(self, val: Optional[int]):
        self.min_block = val

    @classmethod
    def get_competition_parameters(cls, competition_id: str) -> Optional[CompetitionParameters]:
        """Retrieve competition parameters by competition ID."""
        return next((x for x in COMPETITION_SCHEDULE if x.competition_id == competition_id), None)

    async def _get_metadata(self, hotkey: str) -> Optional[ModelMetadata]:
        """Retrieve metadata about a model by hotkey."""
        return await self.metadata_store.retrieve_model_metadata(hotkey)

    async def sync_models(self, hotkeys: list[str]):
        """Synchronize models for a list of hotkeys."""
        tasks = [self.sync_model(hotkey) for hotkey in hotkeys]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def sync_model_metadata_only(self, hotkey: str) -> bool:
        """Update metadata only for a hotkey if out of sync and return whether it was updated."""
        metadata = await self._get_metadata(hotkey)
        if not metadata:
            bt.logging.trace(f"No valid metadata found on the chain for hotkey {hotkey}")
            return False

        if self.min_block and metadata.block < self.min_block:
            bt.logging.trace(f"Skipping model for {hotkey} due to block {metadata.block} being less than minimum block {self.min_block}")
            return False

        # Backwards compatibility for models submitted before competition ID was added
        if not metadata.id.competition_id:
            metadata.id.competition_id = ORIGINAL_COMPETITION_ID

        parameters = self.get_competition_parameters(metadata.id.competition_id)
        if not parameters:
            bt.logging.trace(f"No competition parameters found for {metadata.id.competition_id}")
            return False

        self.model_tracker.on_miner_model_updated_metadata_only(hotkey, metadata)
        return True

    async def ensure_model_downloaded(self, hotkey: str):
        """Ensure that the model for the given hotkey is downloaded."""
        with self.model_tracker.lock:
            if hotkey in self.model_tracker.model_downloaded:
                return

            metadata = self.model_tracker.miner_hotkey_to_model_metadata_dict[hotkey]
            parameters = self.get_competition_parameters(metadata.id.competition_id)

            # Download the new model based on the metadata.
            path = self.local_store.get_path(hotkey)
            model = await self.remote_store.download_model(metadata.id, path, parameters)

            # Validate the hash of the downloaded content.
            if model.id.hash != metadata.id.hash:
                hash_with_hotkey = get_hash_of_two_strings(model.id.hash, hotkey)
                if hash_with_hotkey != metadata.id.hash:
                    bt.logging.trace(
                        f"Sync for hotkey {hotkey} failed. Hash of content downloaded ({model.id.hash}) or hash including hotkey ({hash_with_hotkey}) "
                        f"does not match chain metadata {metadata}."
                    )
                    raise ValueError(
                        f"Sync for hotkey {hotkey} failed. Hash of content downloaded from remote source does not match chain metadata. {metadata}"
                    )

            self.model_tracker.model_downloaded.add(hotkey)

    async def sync_model(self, hotkey: str) -> bool:
        """Update the local model for a hotkey if out of sync and return whether it was updated."""
        metadata = await self._get_metadata(hotkey)
        if not metadata:
            bt.logging.trace(f"No valid metadata found on the chain for hotkey {hotkey}")
            return False

        if self.min_block and metadata.block < self.min_block:
            bt.logging.trace(f"Skipping model for {hotkey} due to block {metadata.block} being less than minimum block {self.min_block}")
            return False

        # Backwards compatibility for models submitted before competition ID was added
        if not metadata.id.competition_id:
            metadata.id.competition_id = ORIGINAL_COMPETITION_ID

        parameters = self.get_competition_parameters(metadata.id.competition_id)
        if not parameters:
            bt.logging.trace(f"No competition parameters found for {metadata.id.competition_id}")
            return False

        # Check what model metadata the tracker currently has for this hotkey.
        tracker_model_metadata = self.model_tracker.get_model_metadata_for_miner_hotkey(hotkey)
        if metadata == tracker_model_metadata:
            return False

        # Download the new model based on the metadata.
        path = self.local_store.get_path(hotkey)
        model = await self.remote_store.download_model(metadata.id, path, parameters)

        # Validate the hash of the downloaded content.
        hash_matches_directly = model.id.hash == metadata.id.hash
        hash_with_hotkey = get_hash_of_two_strings(model.id.hash, hotkey)
        if not (hash_matches_directly or hash_with_hotkey == metadata.id.hash):
            raise ValueError(
                f"Sync for hotkey {hotkey} failed. Hash of content downloaded from remote source does not match chain metadata. {metadata}"
            )

        # Update the tracker
        self.model_tracker.on_miner_model_updated(hotkey, metadata)
        return True
