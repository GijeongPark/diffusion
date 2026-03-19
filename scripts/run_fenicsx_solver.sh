#!/usr/bin/env bash
set -euo pipefail

IMAGE="${FENICSX_IMAGE:-dolfinx/dolfinx:stable}"
WORKDIR="/workspace"

docker run --rm \
  -v "$(pwd):${WORKDIR}" \
  -w "${WORKDIR}" \
  "${IMAGE}" \
  python3 -m peh_inverse_design.fenicsx_modal_solver "$@"
