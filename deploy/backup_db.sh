#!/usr/bin/env bash
set -euo pipefail

DB_PATH="${PAPER_TRADE_DB:-/opt/zhitong-trader/data/paper_trade.db}"
BACKUP_DIR="/opt/zhitong-trader/backups"
mkdir -p "$BACKUP_DIR"

if [ -f "$DB_PATH" ]; then
  STAMP=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
  sqlite3 "$DB_PATH" ".backup '${BACKUP_DIR}/paper_trade_${STAMP}.db'"
  gzip -f "${BACKUP_DIR}/paper_trade_${STAMP}.db"
  find "$BACKUP_DIR" -name 'paper_trade_*.db.gz' -mtime +14 -delete
fi
