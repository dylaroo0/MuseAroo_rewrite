# MuseAroo

MuseAroo is an experiment in modular AI music generation. The repository currently contains early engine stubs and extensive design notes.

## Layout

- `docs/` – project documentation and vision docs
- `src/musearoo/` – Python package with core utilities and engine placeholders
- `scripts/` – assorted helper scripts and prototypes
- `tests/` – unit tests (mostly placeholders)

## Running the demo API

A minimal FastAPI server lives in `src/musearoo/main.py`.
Run it with:

```bash
cd src
uvicorn musearoo.main:app --reload
```

Visit `http://localhost:8000` for a welcome message and `/status` for a simple health check.
