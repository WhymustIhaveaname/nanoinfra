#!/usr/bin/env bash
# run.sh — 起两个服务：静态预览页 + GameNGen 推理后端。
#
# 分两个进程是因为它们的依赖装不到一起：推理那边要 numpy<2（上游 requirements 写死），
# 而 nanoinfra 自己用 numpy 2.x。所以各有各的 venv，页面通过 CORS 调后端。
#
#   ./run.sh            起两个
#   ./run.sh page       只起页面（不玩游戏时省一块显卡）
#   ./run.sh game       只起推理后端
#
# 停：kill $(cat outputs/*.pid)

set -u
cd "$(dirname "$0")"
PAGE_PORT=$(cat .port 2>/dev/null || echo 25676)
GAME_PORT=25677
WHAT=${1:-all}
mkdir -p outputs

start_page() {
  nohup ../../.venv/bin/python -m http.server "$PAGE_PORT" --bind 0.0.0.0 \
        > outputs/page.log 2>&1 &
  echo $! > outputs/page.pid
  echo "页面   http://$(hostname -I | awk '{print $1}'):$PAGE_PORT/   (pid $(cat outputs/page.pid))"
}

start_game() {
  # expandable_segments 治显存碎片；上下文 64 帧的 UNet 输入不小，16GB 卡上留点余量
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup gamengen/.venv/bin/python gamengen/doom_ngen_server.py \
          --port "$GAME_PORT" --bind 0.0.0.0 --bench 25 \
          > outputs/gamengen/server.log 2>&1 &
  echo $! > outputs/game.pid
  echo "推理   :$GAME_PORT  (pid $(cat outputs/game.pid))  模型加载+基准约 100 秒，稍等"
}

case "$WHAT" in
  page) start_page ;;
  game) start_game ;;
  all)  start_page; start_game ;;
  *) echo "用法: $0 [all|page|game]"; exit 1 ;;
esac
