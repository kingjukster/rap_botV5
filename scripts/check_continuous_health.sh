#!/usr/bin/env bash
# Health check + optional auto-remediation for rapbot-continuous (cron every ~20 min).
#
# Cron example:
#   */20 * * * * mkdir -p /path/to/rap_botV5/logs && /path/to/rap_botV5/scripts/check_continuous_health.sh >> /path/to/rap_botV5/logs/continuous_cron.log 2>&1
#
# Env:
#   RAPBOT_CRON_REMEDIATE=1          — try to fix issues (default 1). Set 0 for check-only + escalate.
#   RAPBOT_CRON_STALL_MINUTES=180    — DB runs stuck in status=running longer than this → mark failed + restart
#   RAPBOT_CRON_WEBHOOK_URL=        — if set, POST JSON on unresolved problems after remediation
#   RAPBOT_CRON_NO_CHILD_GRACE=40   — minutes with no run_verse_qd child before restart (default 40)

set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
STALL_AFTER="${RAPBOT_CRON_STALL_MINUTES:-180}"
REMEDIATE="${RAPBOT_CRON_REMEDIATE:-1}"
NO_CHILD_GRACE="${RAPBOT_CRON_NO_CHILD_GRACE:-40}"
NO_CHILD_STATE="$LOG_DIR/.cron_no_child_first_seen"
ISSUES_FILE="$(mktemp)"
ESCALATION_FILE="$LOG_DIR/cursor_cron_escalation.txt"

PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
export PATH

cleanup() { rm -f "$ISSUES_FILE"; }
trap cleanup EXIT

log() {
  echo "[$(date -Iseconds)] $*"
}

issue() {
  log "$1"
  echo "$1" >>"$ISSUES_FILE"
}

mysql_exec() {
  docker exec rapbot-mysql mysql -N -urapbot -prapbot rapbot -e "$1" 2>/dev/null
}

start_continuous() {
  log "REMEDIATE: starting rapbot-continuous via docker compose"
  (cd "$REPO_ROOT" && docker compose run -d --name rapbot-continuous --rm evolution \
    python scripts/run_continuous.py --config config/continuous_runs.yaml --no-docker)
}

restart_continuous() {
  log "REMEDIATE: docker restart rapbot-continuous"
  docker restart rapbot-continuous 2>/dev/null || {
    log "REMEDIATE: restart failed; trying stop + start"
    docker rm -f rapbot-continuous 2>/dev/null || true
    start_continuous
  }
}

mark_stale_running_failed() {
  local th="$1"
  [[ "$th" =~ ^[0-9]+$ ]] && [[ "$th" -gt 0 ]] || return 0
  log "REMEDIATE: marking runs stale ≥${th}m in DB as failed"
  mysql_exec "UPDATE runs SET status='failed' WHERE status='running' AND TIMESTAMPDIFF(MINUTE, updated_at, NOW()) >= ${th};"
}

post_webhook_if_set() {
  [[ -n "${RAPBOT_CRON_WEBHOOK_URL:-}" ]] || return 0
  local summary
  summary="$(paste -sd '; ' "$ISSUES_FILE" 2>/dev/null || true)"
  python3 - "$RAPBOT_CRON_WEBHOOK_URL" "$summary" "$REPO_ROOT" <<'PY' || log "WARN: webhook failed"
import json, sys, urllib.request
url, summary, repo = sys.argv[1], sys.argv[2], sys.argv[3]
body = json.dumps({
    "source": "rapbot-cron",
    "severity": "needs_attention",
    "summary": summary,
    "repo": repo,
}).encode()
req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    urllib.request.urlopen(req, timeout=30)
except Exception as e:
    sys.stderr.write(str(e) + "\n")
    sys.exit(1)
PY
}

write_cursor_escalation() {
  {
    echo "Rap bot continuous runner — cron could not fix all problems"
    echo "Time: $(date -Iseconds)"
    echo "Host: $(hostname)"
    echo "Repo: $REPO_ROOT"
    echo ""
    echo "Issues:"
    cat "$ISSUES_FILE" 2>/dev/null || true
    echo ""
    echo "---"
    echo "Copy everything below into Cursor chat and ask the agent to diagnose and fix:"
    echo ""
    echo "My rap_botV5 cron health check reported problems after auto-remediation."
    echo "Repo: $REPO_ROOT"
    echo "Check: logs/continuous_cron.log , docker logs rapbot-continuous , MySQL runs with status=running."
    echo "Issues summarized:"
    sed 's/^/  - /' "$ISSUES_FILE" 2>/dev/null || true
  } >"$ESCALATION_FILE"
  log "ESCALATE: wrote $ESCALATION_FILE (paste into Cursor if problems remain)"
}

# --- checks-only: sets globals last_rc, populates ISSUES_FILE ---
do_checks() {
  : >"$ISSUES_FILE"
  last_rc=0

  if ! docker info >/dev/null 2>&1; then
    issue "ERROR: docker daemon not reachable"
    last_rc=1
    return
  fi

  mysql_up=false
  if docker ps --filter "name=^rapbot-mysql$" --format '{{.Names}}' 2>/dev/null | grep -q 'rapbot-mysql'; then
    log "OK: rapbot-mysql container is up"
    mysql_up=true
  elif docker ps -a --filter "name=^rapbot-mysql$" --format '{{.Names}}' 2>/dev/null | grep -q 'rapbot-mysql'; then
    issue "WARN: rapbot-mysql exists but is not running"
    last_rc=1
  else
    issue "ERROR: rapbot-mysql container missing"
    last_rc=1
  fi

  continuous_up=false
  if docker ps --filter "name=^rapbot-continuous$" --format '{{.Names}}' 2>/dev/null | grep -q 'rapbot-continuous'; then
    log "OK: rapbot-continuous container is up"
    continuous_up=true
  elif docker ps -a --filter "name=^rapbot-continuous$" --format '{{.Names}}' 2>/dev/null | grep -q 'rapbot-continuous'; then
    issue "WARN: rapbot-continuous exists but is not running"
    last_rc=1
  else
    issue "ERROR: rapbot-continuous container missing"
    last_rc=1
  fi

  pid_ok=false
  if [[ "$continuous_up" == true ]]; then
    pid1="$(docker exec rapbot-continuous sh -c 'tr "\0" " " < /proc/1/cmdline 2>/dev/null || true')"
    if echo "$pid1" | grep -q 'run_continuous'; then
      log "OK: PID1 is run_continuous.py"
      pid_ok=true
    else
      issue "WARN: PID1 not run_continuous: ${pid1:0:120}"
      last_rc=1
    fi
  fi

  child_state="unknown"
  if [[ "$continuous_up" == true ]]; then
    child_state="$(
      docker exec rapbot-continuous sh -c '
        ok=""
        for p in /proc/[0-9]*; do
          [ -r "$p/cmdline" ] || continue
          c=$(tr "\0" " " < "$p/cmdline" 2>/dev/null || true)
          case "$c" in
            *run_verse_qd.py*) ok=1; break ;;
          esac
        done
        if [ -n "$ok" ]; then echo active; else echo none; fi
      ' 2>/dev/null || echo unknown
    )"
  fi

  if [[ "$child_state" == active ]]; then
    log "OK: run_verse_qd.py child process present"
    rm -f "$NO_CHILD_STATE"
  elif [[ "$child_state" == none ]] && [[ "$continuous_up" == true ]]; then
    issue "WARN: no run_verse_qd.py child"
    last_rc=1
  elif [[ "$child_state" == unknown ]] && [[ "$continuous_up" == true ]]; then
    issue "WARN: could not inspect processes inside container"
    last_rc=1
  fi

  stall_age=""
  if [[ "$mysql_up" == true ]]; then
    stall_sql="SELECT COALESCE(MAX(TIMESTAMPDIFF(MINUTE, updated_at, NOW())), -1) FROM runs WHERE status='running';"
    stall_age="$(mysql_exec "$stall_sql" | tr -d '\r' || echo "")"
    if [[ "$stall_age" =~ ^-?[0-9]+$ ]]; then
      if [[ "$stall_age" -lt 0 ]]; then
        log "OK: no runs with status=running in DB"
      elif [[ "$STALL_AFTER" =~ ^[0-9]+$ ]] && [[ "$STALL_AFTER" -gt 0 ]] && [[ "$stall_age" -ge "$STALL_AFTER" ]]; then
        issue "WARN: DB running runs max staleness ${stall_age} min (threshold ${STALL_AFTER})"
        last_rc=1
      else
        log "OK: running run(s) max staleness ${stall_age} min (threshold ${STALL_AFTER})"
      fi
    else
      issue "WARN: could not read DB staleness (got '${stall_age}')"
      last_rc=1
    fi
  fi
}

# --- remediation ---
try_remediate() {
  [[ "$REMEDIATE" == "1" ]] || return 0

  # MySQL container stopped?
  if docker ps -a --filter "name=^rapbot-mysql$" --format '{{.Names}} {{.Status}}' 2>/dev/null | grep -q 'rapbot-mysql'; then
    if ! docker ps --filter "name=^rapbot-mysql$" --format '{{.Names}}' 2>/dev/null | grep -q 'rapbot-mysql'; then
      log "REMEDIATE: docker start rapbot-mysql"
      docker start rapbot-mysql 2>/dev/null || true
      sleep 5
    fi
  fi

  # Stuck runs in DB
  if docker ps --filter "name=^rapbot-mysql$" --format '{{.Names}}' 2>/dev/null | grep -q 'rapbot-mysql'; then
    if [[ "$STALL_AFTER" =~ ^[0-9]+$ ]] && [[ "$STALL_AFTER" -gt 0 ]]; then
      stall_sql="SELECT COALESCE(MAX(TIMESTAMPDIFF(MINUTE, updated_at, NOW())), -1) FROM runs WHERE status='running';"
      age="$(mysql_exec "$stall_sql" | tr -d '\r' || echo -1)"
      if [[ "$age" =~ ^[0-9]+$ ]] && [[ "$age" -ge "$STALL_AFTER" ]]; then
        mark_stale_running_failed "$STALL_AFTER"
        if docker ps --filter "name=^rapbot-continuous$" --format '{{.Names}}' 2>/dev/null | grep -q 'rapbot-continuous'; then
          restart_continuous
          sleep 8
        fi
      fi
    fi
  fi

  # Continuous container missing or not running
  if ! docker ps --filter "name=^rapbot-continuous$" --format '{{.Names}}' 2>/dev/null | grep -q 'rapbot-continuous'; then
    docker rm -f rapbot-continuous 2>/dev/null || true
    start_continuous
    sleep 8
    return 0
  fi

  # PID1 wrong → restart
  pid1="$(docker exec rapbot-continuous sh -c 'tr "\0" " " < /proc/1/cmdline 2>/dev/null || true')"
  if ! echo "$pid1" | grep -q 'run_continuous'; then
    restart_continuous
    sleep 8
    return 0
  fi

  # No child: only restart if grace elapsed (avoid restarting between normal run boundaries)
  child_has=""
  child_has="$(
    docker exec rapbot-continuous sh -c '
      for p in /proc/[0-9]*; do
        [ -r "$p/cmdline" ] || continue
        c=$(tr "\0" " " < "$p/cmdline" 2>/dev/null || true)
        case "$c" in
          *run_verse_qd.py*) echo yes; exit 0 ;;
        esac
      done
      echo no
    ' 2>/dev/null || echo no
  )"
  if [[ "$child_has" != yes ]]; then
    now="$(date +%s)"
    if [[ ! -f "$NO_CHILD_STATE" ]]; then
      echo "$now" >"$NO_CHILD_STATE"
      log "REMEDIATE: no evolution child yet; will restart if still missing for ${NO_CHILD_GRACE}m"
    else
      first="$(cat "$NO_CHILD_STATE" 2>/dev/null || echo "$now")"
      if [[ "$first" =~ ^[0-9]+$ ]]; then
        elapsed=$(( (now - first) / 60 ))
        if [[ "$elapsed" -ge "$NO_CHILD_GRACE" ]]; then
          log "REMEDIATE: no child for ${elapsed}m (grace ${NO_CHILD_GRACE}m) → restart"
          restart_continuous
          rm -f "$NO_CHILD_STATE"
          sleep 8
        else
          log "REMEDIATE: no child; grace ${NO_CHILD_GRACE}m (elapsed ${elapsed}m)"
        fi
      fi
    fi
  fi
}

last_rc=0
do_checks

if [[ "$last_rc" -ne 0 ]]; then
  try_remediate
  do_checks
fi

if [[ "$last_rc" -ne 0 ]]; then
  write_cursor_escalation
  post_webhook_if_set
fi

exit "$last_rc"
