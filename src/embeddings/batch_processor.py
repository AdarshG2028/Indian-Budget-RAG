import json
import logging
from pathlib import Path
from typing import Iterator, List

from .models import ChunkData

logger = logging.getLogger(__name__)

class ChunkBatchProcessor:
    """
    Handles lazy loading and batching of JSON chunks from the filesystem.
    Avoids loading all chunks into memory simultaneously.
    """
    
    def __init__(self, data_dir: Path, batch_size: int = 32):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        
        if not self.data_dir.exists() or not self.data_dir.is_dir():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_dir}")

    def iter_chunks(self) -> Iterator[ChunkData]:
        """Iterates through every chunks.json file and yields ChunkData objects one by one."""
        # Find all chunks.json files recursively
        chunk_files = list(self.data_dir.rglob("chunks.json"))
        
        if not chunk_files:
            logger.warning(f"No chunks.json files found in {self.data_dir}")
            return
            
        logger.info(f"Found {len(chunk_files)} chunks.json files to process.")
        
        for file_path in chunk_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                if not isinstance(data, list):
                    logger.warning(f"Expected a list of chunks in {file_path}, got {type(data)}. Skipping.")
                    continue
                    
                for raw_chunk in data:
                    try:
                        yield ChunkData.from_dict(raw_chunk)
                    except Exception as e:
                        logger.warning(f"Failed to parse a chunk in {file_path}: {e}")
                        
            except json.JSONDecodeError as e:
                logger.error(f"Malformed JSON in {file_path}: {e}. Skipping file.")
            except Exception as e:
                logger.error(f"Unexpected error reading {file_path}: {e}. Skipping file.")

    def iter_batches(self) -> Iterator[List[ChunkData]]:
        """Yields chunks grouped into lists of size self.batch_size."""
        current_batch = []
        for chunk in self.iter_chunks():
            # Skip completely empty chunks
            if not chunk.text or not chunk.text.strip():
                logger.debug(f"Skipping empty chunk: {chunk.chunk_id}")
                continue
                
            current_batch.append(chunk)
            
            if len(current_batch) == self.batch_size:
                yield current_batch
                current_batch = []
                
        # Yield the remainder if any
        if current_batch:
            yield current_batch
