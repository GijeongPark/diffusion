#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
#  PEH Inverse Design — 전체 파이프라인 자동 실행 스크립트
#
#  사용법:
#    bash scripts/run_all.sh
#    bash scripts/run_all.sh --unit-cell-npz data/test_runs/test3/unit_cell_dataset.npz --limit 3 --run-name test3
# ──────────────────────────────────────────────────────────────────────

cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd)"

# ── 옵션 파싱 ──
LIMIT=""
UNIT_CELL_NPZ=""
RUN_NAME=""
OUTPUT_ROOT="runs"
IMAGE="${FENICSX_IMAGE:-dolfinx/dolfinx:stable}"
SUBSTRATE_RHO="7930.0"
PIEZO_RHO="7800.0"

while [ $# -gt 0 ]; do
  case "$1" in
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --limit=*)
      LIMIT="${1#*=}"
      shift
      ;;
    --unit-cell-npz)
      UNIT_CELL_NPZ="$2"
      shift 2
      ;;
    --unit-cell-npz=*)
      UNIT_CELL_NPZ="${1#*=}"
      shift
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --run-name=*)
      RUN_NAME="${1#*=}"
      shift
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --output-root=*)
      OUTPUT_ROOT="${1#*=}"
      shift
      ;;
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --image=*)
      IMAGE="${1#*=}"
      shift
      ;;
    --substrate-rho)
      SUBSTRATE_RHO="$2"
      shift 2
      ;;
    --substrate-rho=*)
      SUBSTRATE_RHO="${1#*=}"
      shift
      ;;
    --piezo-rho)
      PIEZO_RHO="$2"
      shift 2
      ;;
    --piezo-rho=*)
      PIEZO_RHO="${1#*=}"
      shift
      ;;
    *)
      echo "ERROR: unknown argument: $1"
      exit 1
      ;;
  esac
done

VENV_PYTHON="./.venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
  echo "ERROR: .venv not found. Run: python -m venv .venv && .venv/bin/pip install -e ."
  exit 1
fi

if [ -z "$UNIT_CELL_NPZ" ]; then
  if [ -f "data/unit_cell_dataset.npz" ]; then
    UNIT_CELL_NPZ="data/unit_cell_dataset.npz"
  else
    UNIT_CELL_NPZ="data/dataset_100.npz"
  fi
fi

if [ ! -f "$UNIT_CELL_NPZ" ]; then
  echo "ERROR: unit-cell dataset not found: $UNIT_CELL_NPZ"
  exit 1
fi

if [ -z "$RUN_NAME" ]; then
  if [ -n "$LIMIT" ]; then
    RUN_NAME=$(printf "test_n%03d" "$LIMIT")
  else
    RUN_NAME="$(basename "$UNIT_CELL_NPZ" .npz)"
  fi
fi

RUN_ROOT="${OUTPUT_ROOT}/${RUN_NAME}"
MESH_DIR="${RUN_ROOT}/meshes/volumes"
RESPONSE_DIR="${RUN_ROOT}/data/fem_responses"
MODES_DIR="${RUN_ROOT}/data/modal_data"
RESPONSE_OUTPUT="${RUN_ROOT}/data/response_dataset.npz"
REPORT_DIR="${RUN_ROOT}/reports"

echo "Run name:         $RUN_NAME"
echo "Unit-cell input:  $UNIT_CELL_NPZ"
echo "Run root:         $RUN_ROOT"
echo "Substrate rho:    $SUBSTRATE_RHO"
echo "Piezo rho:        $PIEZO_RHO"

# ══════════════════════════════════════════════════════════════════════
INTEGRATED_OUTPUT="${RUN_ROOT}/data/integrated_dataset.npz"

echo ""
echo "===== Step 1/6: 3D 볼륨 메쉬 생성 ====="
echo ""

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
  LIMIT_ARG="--limit $LIMIT"
fi

$VENV_PYTHON -m peh_inverse_design.build_volume_meshes \
  --unit-cell-npz "$UNIT_CELL_NPZ" \
  --mesh-dir "$MESH_DIR" \
  $LIMIT_ARG

# ══════════════════════════════════════════════════════════════════════
echo ""
echo "===== Step 2/6: Docker 이미지 확인 ====="
echo ""

if ! command -v docker &>/dev/null; then
  echo "ERROR: Docker가 설치되어 있지 않습니다."
  echo "  https://docs.docker.com/get-docker/ 에서 설치하세요."
  exit 1
fi

if ! docker image inspect "$IMAGE" &>/dev/null; then
  echo "Docker 이미지 다운로드 중: $IMAGE"
  docker pull "$IMAGE"
fi
echo "Docker 이미지 준비 완료: $IMAGE"

# ══════════════════════════════════════════════════════════════════════
echo ""
echo "===== Step 3/6: FEniCSx 솔버 실행 ====="
echo ""

mkdir -p "$RESPONSE_DIR" "$MODES_DIR"

MESH_FILES=($(ls "$MESH_DIR"/plate3d_*_fenicsx.npz 2>/dev/null | sort))
TOTAL=${#MESH_FILES[@]}

if [ "$TOTAL" -eq 0 ]; then
  echo "ERROR: $MESH_DIR 에 메쉬 파일이 없습니다."
  exit 1
fi

OK=0
FAIL=0
for i in "${!MESH_FILES[@]}"; do
  MESH_FILE="${MESH_FILES[$i]}"
  MESH_BASENAME=$(basename "$MESH_FILE")
  IDX=$((i + 1))
  echo -n "[$IDX/$TOTAL] $MESH_BASENAME ... "

  if docker run --rm \
    -v "${PROJECT_DIR}:/workspace" \
    -w "/workspace" \
    "$IMAGE" \
    python3 -m peh_inverse_design.fenicsx_modal_solver \
      --mesh "/workspace/${MESH_FILE}" \
      --response-dir "/workspace/${RESPONSE_DIR}" \
      --modes-dir "/workspace/${MODES_DIR}" \
      --substrate-rho "$SUBSTRATE_RHO" \
      --piezo-rho "$PIEZO_RHO" \
    2>&1; then
    OK=$((OK + 1))
    echo "OK"
  else
    FAIL=$((FAIL + 1))
    echo "FAIL"
  fi
done

echo ""
echo "FEM 완료: 성공=$OK 실패=$FAIL / 총=$TOTAL"

# ══════════════════════════════════════════════════════════════════════
echo ""
echo "===== Step 4/6: 응답 데이터 합치기 ====="
echo ""

$VENV_PYTHON -m peh_inverse_design.build_response_dataset \
  --response-dir "$RESPONSE_DIR" \
  --output "$RESPONSE_OUTPUT"

# ══════════════════════════════════════════════════════════════════════
echo ""
echo "===== Step 5/6: 통합 데이터셋 생성 ====="
echo ""

INTEGRATED_LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
  INTEGRATED_LIMIT_ARG="--limit $LIMIT"
fi

$VENV_PYTHON -m peh_inverse_design.build_integrated_dataset \
  --unit-cell-npz "$UNIT_CELL_NPZ" \
  --response-dir "$RESPONSE_DIR" \
  --modal-dir "$MODES_DIR" \
  --mesh-dir "$MESH_DIR" \
  --output "$INTEGRATED_OUTPUT" \
  $INTEGRATED_LIMIT_ARG

# ══════════════════════════════════════════════════════════════════════
echo ""
echo "===== Step 6/6: 결과 시각화 ====="
echo ""

MPLCONFIGDIR=/tmp/mpl $VENV_PYTHON peh_inverse_design/visualize_run_outputs.py \
  --dataset "$UNIT_CELL_NPZ" \
  --mesh-dir "$MESH_DIR" \
  --response-dir "$RESPONSE_DIR" \
  --modal-dir "$MODES_DIR" \
  --output-dir "$REPORT_DIR"

echo ""
echo "===== 완료! ====="
echo "  기하 데이터:  $MESH_DIR/"
echo "  ANSYS 입력:   ${MESH_DIR}/plate3d_*_ansys.inp"
echo "  FEM 응답:     $RESPONSE_DIR/"
echo "  모달 데이터:  $MODES_DIR/"
echo "  최종 데이터셋: $RESPONSE_OUTPUT"
echo "  통합 데이터셋: $INTEGRATED_OUTPUT"
echo "  리포트:       $REPORT_DIR/"
