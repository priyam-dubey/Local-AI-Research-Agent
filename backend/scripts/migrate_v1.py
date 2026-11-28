"""
Import documents from a v1 install into the new workspace system.

v1 kept the original uploads in data/uploads and a pickled global FAISS index. The old index cannot be
converted (different format, no page metadata), but the original files can simply be re-indexed:

    python -m scripts.migrate_v1 --uploads ../data/uploads [--workspace default]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from app.config import settings
from app.core.errors import UnsupportedTypeError
from app.core.security import safe_filename, unique_upload_path
from app.rag.loaders import detect_type
from app.services import document_service, repo


def migrate(uploads: Path, workspace: str = repo.DEFAULT_WORKSPACE_ID) -> dict:
    repo.init_db()
    repo.get_workspace(workspace)
    done, skipped, failed = [], [], []
    for src in sorted(p for p in uploads.iterdir() if p.is_file()):
        try:
            ftype = detect_type(src.name)
        except UnsupportedTypeError:
            skipped.append(src.name)
            continue
        tmp = unique_upload_path(settings.UPLOAD_DIR, src.name)
        shutil.copyfile(src, tmp)                                # process_document consumes (deletes) its temp file
        doc = repo.create_document(workspace, safe_filename(src.name), ftype, src.stat().st_size)
        document_service.process_document(doc, tmp)
        row = repo.get_document(workspace, doc["id"])
        (done if row["status"] == "ready" else failed).append(src.name if row["status"] == "ready" else f"{src.name}: {row['error']}")
    return {"imported": done, "skipped": skipped, "failed": failed}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uploads", type=Path, required=True)
    ap.add_argument("--workspace", default=repo.DEFAULT_WORKSPACE_ID)
    a = ap.parse_args()
    r = migrate(a.uploads, a.workspace)
    print(f"Imported {len(r['imported'])}, skipped {len(r['skipped'])} (unsupported), failed {len(r['failed'])}")
    for f in r["failed"]:
        print("  failed:", f)


if __name__ == "__main__":
    main()
