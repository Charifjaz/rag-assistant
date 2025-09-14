from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from loguru import logger
import time
import os
import config

# Configuration des logs
logger.add("logs/detection.log", rotation="1 MB", retention="7 days", level="INFO")

# Handler personnalisé pour surveiller les fichiers ajoutés
class PDFDetectionHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".pdf"):
            file_name = os.path.basename(event.src_path)
            logger.info(f"📄 Nouveau PDF détecté : {file_name} (chemin : {event.src_path})")

# Dossier à surveiller
WATCH_DIR = config.DATA_FOLDER_AVOCAT

if __name__ == "__main__":
    logger.info(f"🕵️ Surveillance du dossier : {WATCH_DIR}")
    
    event_handler = PDFDetectionHandler()
    observer = Observer()
    observer.schedule(event_handler, path=WATCH_DIR, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("🛑 Arrêt de la surveillance (Ctrl+C)")
        observer.stop()

    observer.join()
