import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent
from typing import Set
from app.config import settings, logger
from app.indexer import Indexer


class NoteFileHandler(FileSystemEventHandler):
    """Xử lý sự kiện thay đổi file trong thư mục notes."""

    def __init__(self, indexer: Indexer):
        super().__init__()
        self.indexer = indexer
        self._debounce_set: Set[str] = set()
        self._debounce_delay = 2.0

    def _should_process(self, path: str) -> bool:
        path_obj = Path(path)
        if path_obj.suffix.lower() != ".md":
            return False
        if path_obj.name.lower() == "template.md":
            return False
        if any(part.startswith(".") for part in path_obj.parts):
            return False
        return True

    def _debounced_index(self, filepath: Path):
        path_str = str(filepath)
        if path_str in self._debounce_set:
            return
        self._debounce_set.add(path_str)
        time.sleep(self._debounce_delay)
        try:
            if filepath.exists():
                logger.info(f"🔄 Re-indexing: {filepath.name}")
                self.indexer.index_file(filepath)
            else:
                logger.info(f"🗑️ File removed: {filepath.name}")
        except Exception as e:
            logger.error(f"Failed to index {filepath}: {e}")
        finally:
            self._debounce_set.discard(path_str)

    def on_modified(self, event):
        if isinstance(event, FileModifiedEvent) and self._should_process(event.src_path):
            self._debounced_index(Path(event.src_path))

    def on_created(self, event):
        if isinstance(event, FileCreatedEvent) and self._should_process(event.src_path):
            logger.info(f"📄 New note: {Path(event.src_path).name}")
            self._debounced_index(Path(event.src_path))

    def on_deleted(self, event):
        if isinstance(event, FileDeletedEvent) and self._should_process(event.src_path):
            logger.info(f"🗑️ Note deleted: {Path(event.src_path).name}")


class FileWatcher:
    """Wrapper quản lý Observer watchdog."""

    def __init__(self, indexer: Indexer = None, notes_dir: Path = None):
        self.notes_dir = notes_dir or settings.NOTES_DIR
        self.indexer = indexer or Indexer()
        self.observer = Observer()
        self.handler = NoteFileHandler(self.indexer)
        self._running = False

    def start(self):
        self.observer.schedule(self.handler, str(self.notes_dir), recursive=True)
        self.observer.start()
        self._running = True
        logger.info(f"👀 Watching: {self.notes_dir}")

    def stop(self):
        self._running = False
        self.observer.stop()
        self.observer.join()
        logger.info("🛑 File watcher stopped")

    def health_check(self) -> bool:
        return self._running and self.observer.is_alive()


if __name__ == "__main__":
    logger.info("🚀 Starting DevMemory File Watcher standalone...")
    watcher = FileWatcher()
    watcher.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        watcher.stop()
