#!/usr/bin/env bash
# Run PyFlayer from the source folder (after ./install.sh). Default: desktop app.
#   ./run.sh            -> GUI
#   ./run.sh run        -> console (server + bot + AI)
#   ./run.sh train      -> 7 hour overnight training
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=.venv/bin/python
[ -x "$PY" ] || PY=python3
exec "$PY" -m pyflayer "${@:-gui}"
