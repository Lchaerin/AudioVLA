#!/usr/bin/env bash
# Download pretrained model checkpoints for Audio-VLA
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CKPT_DIR="$PROJECT_ROOT/checkpoints"
mkdir -p "$CKPT_DIR"

# ── LAION CLAP ────────────────────────────────────────────────────────────────
# laion-clap 패키지를 Python에서 import하면 AudioVLA의 training/ 폴더와
# 이름 충돌이 발생하므로, HuggingFace Hub에서 직접 wget으로 다운로드합니다.
CLAP_FILE="music_speech_audioset_epoch_15_esc_89.98.pt"
CLAP_URL="https://huggingface.co/lukewys/laion_clap/resolve/main/${CLAP_FILE}"

if [ -f "$CKPT_DIR/$CLAP_FILE" ]; then
    echo "CLAP 체크포인트 이미 존재: $CKPT_DIR/$CLAP_FILE"
else
    echo "LAION CLAP 체크포인트 다운로드 중..."
    echo "  URL: $CLAP_URL"
    wget -q --show-progress -O "$CKPT_DIR/$CLAP_FILE" "$CLAP_URL" \
        || curl -L --progress-bar -o "$CKPT_DIR/$CLAP_FILE" "$CLAP_URL"
    echo "  저장 완료: $CKPT_DIR/$CLAP_FILE"
fi

# ── SmolVLA via HuggingFace Hub ───────────────────────────────────────────────
# lerobot 대신 huggingface_hub으로 직접 다운로드 (충돌 없음)
SMOLVLA_DIR="$CKPT_DIR/smolvla_base"

if [ -d "$SMOLVLA_DIR" ] && [ "$(ls -A "$SMOLVLA_DIR" 2>/dev/null)" ]; then
    echo "SmolVLA 체크포인트 이미 존재: $SMOLVLA_DIR"
else
    echo "SmolVLA 체크포인트 다운로드 중 (lerobot/smolvla_base)..."
    # PROJECT_ROOT 밖 임시 디렉터리에서 실행 → training/ 이름 충돌 방지
    python - <<EOF
import sys, os, tempfile
# 임시 디렉터리를 작업 경로로 사용해 로컬 training/ 모듈 충돌 방지
os.chdir(tempfile.gettempdir())

try:
    from huggingface_hub import snapshot_download
    local = snapshot_download(
        repo_id="lerobot/smolvla_base",
        local_dir="$SMOLVLA_DIR",
        ignore_patterns=["*.bin"],   # safetensors 선호
    )
    print(f"SmolVLA 저장 완료: {local}")
except Exception as e:
    print(f"huggingface_hub 다운로드 실패: {e}")
    print("대신 lerobot API 시도...")
    try:
        sys.path.insert(0, "$PROJECT_ROOT")
        os.chdir("$CKPT_DIR")          # training/ 없는 디렉터리
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        policy.save_pretrained("$SMOLVLA_DIR")
        print("SmolVLA 저장 완료: $SMOLVLA_DIR")
    except Exception as e2:
        print(f"lerobot 다운로드도 실패: {e2}")
        print("수동 다운로드: huggingface-cli download lerobot/smolvla_base")
        sys.exit(1)
EOF
fi

# ── configs/default.yaml 체크포인트 경로 안내 ─────────────────────────────────
echo ""
echo "=============================="
echo " 다운로드 완료"
echo "=============================="
echo "  CLAP    : $CKPT_DIR/$CLAP_FILE"
echo "  SmolVLA : $SMOLVLA_DIR"
echo ""
echo "configs/default.yaml 에서 경로를 확인하세요:"
echo "  checkpoints:"
echo "    clap_checkpoint: $CKPT_DIR/$CLAP_FILE"
echo "    smolvla_checkpoint: $SMOLVLA_DIR"
