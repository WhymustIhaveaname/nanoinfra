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
# 对外的 25676 现在归 nginx（HTTPS + /api 反代，见 DEPLOY.md）。这里这个 python
# 静态服务器只剩本机自用：不想动 nginx、或者 nginx 没起的时候拿它看页面。
# 绑 127.0.0.1 是故意的——对外一律走 nginx，别再开一个明文口子。
PAGE_PORT=$(cat .port 2>/dev/null || echo 25675)
GAME_PORT=25677
WHAT=${1:-all}
mkdir -p outputs outputs/gamengen

start_page() {
  nohup ../../.venv/bin/python -m http.server "$PAGE_PORT" --bind 127.0.0.1 \
        > outputs/page.log 2>&1 &
  echo $! > outputs/page.pid
  echo "页面   http://127.0.0.1:$PAGE_PORT/  (仅本机；对外是 https://<host>:25676/，nginx 常驻)"
}

start_game() {
  # expandable_segments 治显存碎片；上下文 64 帧的 UNet 输入不小，16GB 卡上留点余量
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup gamengen/.venv/bin/python gamengen/doom_ngen_server.py \
          --port "$GAME_PORT" --bind 127.0.0.1 --bench 25 \
          > outputs/gamengen/server.log 2>&1 &
  echo $! > outputs/game.pid
  echo "推理   :$GAME_PORT  (pid $(cat outputs/game.pid))  模型加载+基准约 20 秒，稍等"
}

case "$WHAT" in
  page) start_page ;;
  game) start_game ;;
  all)  start_page; start_game ;;
  *) echo "用法: $0 [all|page|game]"; exit 1 ;;
esac
