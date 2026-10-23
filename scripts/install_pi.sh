#!/usr/bin/env bash
# Install robolab on a headless Ubuntu / Debian machine: Raspberry Pi 4/5 with
# 64-bit Ubuntu Server, Jetson, any arm64 or x86-64 box. No desktop needed.
#
#   bash scripts/install_pi.sh            # install into ~/robolab-venv and run tests
#
# Environment:
#   VENV=/path        where to create the virtual environment (default ~/robolab-venv)
#   PYTHON=python3.x  interpreter to use (default python3)
#   SKIP_APT=1        do not call apt-get (packages already installed / no sudo)
#   SKIP_TESTS=1      do not run the test suite
#   ROBOLAB_NATIVE=0  build a portable binary instead of one tuned for this CPU
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$HOME/robolab-venv}"
PYTHON="${PYTHON:-python3}"
ARCH="$(uname -m)"
OS="$( (. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME") || uname -s)"
MODEL="$ARCH"
[ -r /proc/device-tree/model ] && MODEL="$(tr -d '\0' </proc/device-tree/model)"

echo "== robolab: $MODEL | $OS | $ARCH | $(nproc) cores"
case "$ARCH" in
  aarch64|x86_64) ;;
  armv7l|armv6l)
    echo "!! 32-bit ARM: numpy/scipy have no ready wheels here and will compile for hours."
    echo "!! Install the 64-bit (arm64) Ubuntu Server image for the Pi instead." ;;
esac

if [ "${SKIP_APT:-0}" != "1" ]; then
  SUDO=""; [ "$(id -u)" != "0" ] && SUDO="sudo"
  echo "== system packages: python3-venv python3-dev build-essential"
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq python3-venv python3-dev build-essential >/dev/null
fi

echo "== virtual environment: $VENV ($($PYTHON --version))"
"$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1091
. "$VENV/bin/activate"
python -m pip install -q --upgrade pip wheel

echo "== building robolab (Cython + OpenMP), takes 1-3 min on a Pi..."
export ROBOLAB_NATIVE="${ROBOLAB_NATIVE:-1}"
python -m pip install -v "$HERE[plot,test]" > "$VENV/robolab-build.log" 2>&1 || {
  echo "!! build failed, last lines of $VENV/robolab-build.log:"; tail -30 "$VENV/robolab-build.log"; exit 1; }
grep -E "robolab:" "$VENV/robolab-build.log" | sed 's/^ */   /' || true

if [ "${SKIP_TESTS:-0}" != "1" ]; then
  echo "== tests"
  (cd /tmp && python -m pytest -q -p no:cacheprovider "$HERE/tests" | tail -3)
fi

robolab info --out "$HERE/results"
cat <<MSG

Done. In every new shell:
    source $VENV/bin/activate
Then, for example:
    robolab all --quick          # everything, about 1-3 min on a Pi 4
    robolab walk --gif           # walking simulation with animation
    robolab bench --threads 4
Results (tables, PNG, CSV) go to ./results; to look at them from a laptop:
    python3 -m http.server 8000 --directory results   # open http://<pi-ip>:8000
MSG
