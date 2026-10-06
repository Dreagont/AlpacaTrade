"""Protect trading databases (including aliases and SQLite sidecars) from CSV output."""

from pathlib import Path

import config


def csv_output_path(output, *, db_path=None):
    target = Path(output).resolve()
    databases = {Path(config.DATABASE_PATH).resolve()}
    if db_path is not None:
        databases.add(Path(db_path).resolve())
    for database in databases:
        protected = [database, Path(str(database) + "-wal"), Path(str(database) + "-shm")]
        for path in protected:
            if target == path or (target.exists() and path.exists() and target.samefile(path)):
                raise ValueError("CSV output must not overwrite a trading database or SQLite sidecar")
    return target
