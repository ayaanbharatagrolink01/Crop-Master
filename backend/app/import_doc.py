"""Add regional PDFs (agromet advisories, contingency plans, ICAR guides, pest alerts, groundwater/soil reports)
for the agronomist. With QDRANT_URL set in .env they are also embedded into Qdrant for meaning-based search.

Run from the backend folder.

One file:
  python -m app.import_doc "D:\\pdfs\\vidisha_advisory_07oct.pdf" --title "Vidisha Agromet Advisory, 07 Oct" ^
      --state "Madhya Pradesh" --district Vidisha --crops "soybean, wheat, gram" --valid-until 2026-10-14 ^
      --doc-type advisory
  --district empty  -> the whole state;  --state empty too -> everywhere (e.g. an ICAR crop guide)

A whole folder (files already imported are skipped, so run it again after adding PDFs):
  python -m app.import_doc --folder ..\\data\\regional_docs
    regional_docs\\Madhya Pradesh\\Vidisha\\advisory_07oct.pdf      -> Vidisha district only
    regional_docs\\Madhya Pradesh\\_state\\mp_pest_alert.pdf        -> all of Madhya Pradesh
    regional_docs\\_india\\icar_soybean_ipm.pdf                     -> every farmer
  The file name becomes the title (underscores become spaces).
  Add --replace to re-read files imported before (e.g. after an extraction fix).

Rebuild the Qdrant index from what is already imported (after changing Qdrant or EMBED_MODEL):
  python -m app.import_doc --reindex
Only test the Qdrant connection:
  python -m app.import_doc --check
"""
import argparse
import asyncio
import os
import sys
from urllib.parse import urlparse
from pathlib import Path

from . import rag
from .region import add_document, delete_document, doc_id_by_source, has_source, reindex_all

TYPE_HINTS = {"advisory": "advisory", "agromet": "advisory", "contingency": "contingency_plan",
              "alert": "pest_alert", "pest": "pest_alert", "groundwater": "groundwater", "soil": "soil_report",
              "guide": "guide", "package": "guide", "pop": "guide"}


def _guess_type(name: str) -> str:
    low = name.lower()
    return next((t for k, t in TYPE_HINTS.items() if k in low), "")


async def import_folder(root: Path, replace: bool = False) -> None:
    added = skipped = failed = replaced = 0
    for pdf in sorted(root.rglob("*.pdf")):
        parts = pdf.relative_to(root).parts[:-1]
        if not parts or parts[0] == "_india":
            state, district = "", ""
        elif len(parts) == 1 or parts[1] == "_state":
            state, district = parts[0], ""
        else:
            state, district = parts[0], parts[1]
        source = str(pdf.relative_to(root))
        if has_source(source):
            if not replace:
                skipped += 1
                continue
            await delete_document(doc_id_by_source(source))  # re-read it with the current extractor/chunker
            replaced += 1
        try:
            d = await add_document(pdf.read_bytes(), title=pdf.stem.replace("_", " "), source=source,
                                   state=state, district=district, doc_type=_guess_type(pdf.stem))
            added += 1
            print(f"Added {d['id']}: {source} -> {district or 'all districts'}, {state or 'all states'}"
                  + (f" ({d['indexed_chunks']} chunks in Qdrant)" if "indexed_chunks" in d else ""))
        except Exception as e:  # one bad file must not stop the rest of the folder
            failed += 1
            print(f"Skipped {source}: {type(e).__name__}: {e}")
    print(f"\nDone: {added} added ({replaced} of them replaced), {skipped} already imported, {failed} could not be read")


async def check_qdrant() -> None:
    """Fail early with a plain message instead of a long traceback."""
    raw = os.getenv("QDRANT_URL", "").strip()
    if not raw and os.getenv("QDRANT_PATH"):  # local mode
        try:
            await rag.ensure_collection()
            n = (await rag.qdrant().count(rag.COLLECTION)).count
        except Exception as e:
            hint = (" The folder is in use by another program: stop the uvicorn server (Ctrl+C) and run this again."
                    if "already accessed" in str(e) or "lock" in str(e).lower() else "")
            sys.exit(f"Local Qdrant failed: {type(e).__name__}: {str(e)[:150]}{hint}")
        print(f"OK: local Qdrant in {os.getenv('QDRANT_PATH')}; collection '{rag.COLLECTION}' has {n} chunks")
        return
    if not raw:
        sys.exit("Set QDRANT_URL (a Qdrant server) or QDRANT_PATH (local folder) in backend/.env")
    u = urlparse(raw)
    if raw == ":memory:":  # in-memory test mode
        await rag.ensure_collection()
        print("OK: in-memory Qdrant (test mode)")
        return
    if u.scheme not in ("http", "https") or not u.hostname:
        sys.exit(f"QDRANT_URL must start with https:// and include the port, e.g. https://{raw.split('/')[0]}:443 "
                 f"(now it is {raw[:12]}...)")
    port = u.port or 6333  # qdrant-client uses 6333 when the URL has no port
    prefix = os.getenv("QDRANT_PREFIX", "").strip("/")
    print(f"Connecting to {u.scheme}://{u.hostname}:{port}{'/' + prefix if prefix else ''} ...")
    try:
        await rag.ensure_collection()
        n = (await rag.qdrant().count(rag.COLLECTION)).count
    except Exception as e:
        msg = str(e)
        hint = ("Wrong or read-only QDRANT_API_KEY." if any(x in msg for x in ("401", "403", "Unauthorized", "Forbidden"))
                else "Cannot reach the server: check the port (443 or 6333), VPN/office network, or IP whitelist. "
                     f"Test in a browser: {u.scheme}://{u.hostname}:{port}{'/' + prefix if prefix else ''}/healthz")
        sys.exit(f"Qdrant check failed: {type(e).__name__}: {msg[:120]}\n-> {hint}")
    print(f"OK: connected; collection '{rag.COLLECTION}' has {n} chunks")


async def run(a) -> None:
    if a.check or a.reindex:
        if not rag.enabled():
            sys.exit("Set QDRANT_URL (a Qdrant server) or QDRANT_PATH (local folder) in backend/.env first")
        await check_qdrant()
        if a.check:
            return
        print(f"Re-indexed {await reindex_all()} chunks into Qdrant collection {rag.COLLECTION}")
        return
    if a.folder:
        root = Path(a.folder)
        if not root.is_dir():
            sys.exit(f"Folder not found: {root}")
        await import_folder(root, replace=a.replace)
        return
    try:
        d = await add_document(Path(a.pdf).read_bytes(), title=a.title, source=Path(a.pdf).name, state=a.state,
                               district=a.district, crops=a.crops, valid_until=a.valid_until,
                               doc_type=a.doc_type or _guess_type(a.title))
    except (OSError, ValueError) as e:
        sys.exit(f"Could not add {a.pdf}: {e}")
    print(f"Added {d['id']}: {d['title']} ({d['pages']} pages) for "
          f"{d['district'] or 'all districts'}, {d['state'] or 'all states'}"
          + (f"; {d['indexed_chunks']} chunks in Qdrant" if "indexed_chunks" in d else "; Qdrant not configured"))


def main():
    ap = argparse.ArgumentParser(description="Add regional PDFs for the agronomist")
    ap.add_argument("pdf", nargs="?", help="one PDF file")
    ap.add_argument("--folder", help="import every PDF in this folder (State/District/file.pdf layout)")
    ap.add_argument("--reindex", action="store_true", help="rebuild the Qdrant index from imported documents")
    ap.add_argument("--check", action="store_true", help="only test the connection to Qdrant")
    ap.add_argument("--replace", action="store_true", help="with --folder: re-import files that were imported before")
    ap.add_argument("--title")
    ap.add_argument("--state", default="")
    ap.add_argument("--district", default="")
    ap.add_argument("--crops", default="")
    ap.add_argument("--valid-until", default="", help="YYYY-MM-DD; omit if it never expires")
    ap.add_argument("--doc-type", default="", help="advisory, contingency_plan, pest_alert, guide, groundwater...")
    a = ap.parse_args()
    if not (a.check or a.reindex or a.folder or (a.pdf and a.title)):
        ap.error("give a PDF and --title, or --folder, --reindex or --check")
    asyncio.run(run(a))


if __name__ == "__main__":
    main()