"""Scheduled data retention cleanup job.
Run daily (e.g. via cron, Task Scheduler, or APScheduler) to delete old query history.
Default retention: 90 days.  Users may delete their own data at any time via DELETE /me.
"""
import datetime
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.db import SessionLocal, Query, User, create_tables

logger = logging.getLogger("lic_recommender.cleanup")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RETENTION_DAYS = int(os.environ.get("HISTORY_RETENTION_DAYS", "90"))


def run_cleanup(dry_run: bool = False) -> dict:
    """Delete query history rows older than RETENTION_DAYS. Returns a summary dict."""
    create_tables()
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=RETENTION_DAYS)
    # SQLite stores naive datetimes; normalise by stripping tz for comparison
    cutoff_naive = cutoff.replace(tzinfo=None)

    with SessionLocal() as db:
        old_queries = (
            db.query(Query)
            .filter(Query.created_at < cutoff_naive)
            .all()
        )
        count = len(old_queries)
        if not dry_run:
            for q in old_queries:
                db.delete(q)
            db.commit()
            logger.info("Cleanup removed %d query history rows older than %d days", count, RETENTION_DAYS)
        else:
            logger.info("DRY RUN: would remove %d query rows older than %d days", count, RETENTION_DAYS)

    return {"deleted_queries": count, "dry_run": dry_run, "retention_days": RETENTION_DAYS}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scheduled history retention cleanup")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = run_cleanup(dry_run=args.dry_run)
    print(summary)
