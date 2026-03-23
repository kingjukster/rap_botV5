# Remote Development & Execution Workflow

Use Cursor over SSH to treat the server as a remote dev machine. The web app and evolution run on the server; you edit and run from Cursor.

---

## Prerequisites

- SSH access to the server
- Docker + Docker Compose on the server
- Cursor with Remote-SSH

---

## Step 1 — Connect Cursor to the server

1. `Ctrl + Shift + P` → **Remote-SSH: Connect to Host**
2. Select your server
3. **Open Folder** → `/home/theredia/workspace/evo/rap_botV5` (or `/home/theredia/rap_botV5`)

---

## Step 2 — Start the stack

In a Cursor terminal (runs on the server):

```bash
docker compose up -d
```

This starts:
- **MySQL** on port 3307 (host)
- **Web dashboard** on port 8000 (host)

---

## Step 3 — Access the web app

From your **local browser** (not localhost on the server):

```
http://<SERVER_IP>:8000
```

Example: `http://192.168.68.84:8000`

| Location | `localhost:8000` points to |
|----------|----------------------------|
| Server   | The server itself          |
| Your PC  | Your own computer          |

Use the server IP, not `localhost`, when opening from your PC.

---

## Step 4 — Run evolution

### Web-triggered (Evolve page)

Starting evolution from the web app (`/evolve`) runs jobs in the evolution Docker container. The web container spawns it via the Docker socket. Before first use:

```bash
docker compose build evolution
```

Ensure you run `docker compose up` from the project directory so `RAPBOT_HOST_PROJECT_PATH` (from `${PWD}`) and the compose network are set correctly. If your compose project name differs from `rap_botv5`, set `RAPBOT_DOCKER_NETWORK` and `RAPBOT_EVOLUTION_IMAGE` in the web service environment.

### Option A: Docker (no venv needed)

```bash
docker compose --profile evolution run --rm evolution \
  python scripts/run_verse_evolution.py \
  --theme "pressure,mask,survival" \
  --population 30 \
  --generations 5
```

First run builds the evolution image (~2–5 min); later runs are faster.

### Option B: Python venv (host)

1. Install venv support: `sudo apt install python3.12-venv`
2. Run: `./scripts/setup_venv.sh`
3. Activate: `source venv/bin/activate`
4. Run:

```bash
python scripts/run_verse_evolution.py \
  --theme "pressure,mask,survival" \
  --population 30 \
  --generations 5
```

---

## Architecture

```
[ Cursor IDE ]  ← You edit here
       ↓ SSH
[ Server ]
   ├── Docker Compose
   │     ├── mysql (3307)
   │     ├── web (8000)
   │     └── evolution (on-demand)
   │
   └── Python venv (optional)
         └── evolution scripts
```

---

## Common mistakes

| Mistake | Fix |
|---------|-----|
| Using `localhost:8000` in browser from your PC | Use `http://<SERVER_IP>:8000` |
| Installing packages globally | Use venv or Docker |
| Wrong DB host for host-run scripts | Use `127.0.0.1:3307` in `.env` when running on host |
| Evolution can't reach MySQL | When using Docker evolution, DB host is `mysql` (internal) — already set in compose |

---

## DB config

| Where script runs | `RAPBOT_DB_HOST` |
|-------------------|------------------|
| Host (venv)       | `127.0.0.1`      |
| Docker evolution  | `mysql`          |

The compose `evolution` service sets `RAPBOT_DB_HOST=mysql` automatically. Your `.env` is for host use; override via compose `environment`.
