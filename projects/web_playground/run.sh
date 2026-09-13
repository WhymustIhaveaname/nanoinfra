#!/usr/bin/env bash
# run.sh — 在本机起 GameNGen 推理后端。
#
# 页面不在这里起：静态文件由常驻的 nginx 直读目录（HTTPS 25676，/api 反代到
# 25677），见 DEPLOY.md。原来还有一个绑 127.0.0.1:25675 的 python 静态服务器当
# 后备，2026-09-13 删掉了——nginx 开机常驻，它从来没被用到。
#
# 推理后端单独一个 venv 是因为依赖装不到一起：上游 requirements 写死 numpy<2，
# 而 nanoinfra 自己用 numpy 2.x。
#
#   ./run.sh            起本机推理后端
#
# 停：kill $(cat outputs/game.pid)

set -u
cd "$(dirname "$0")"
GAME_PORT=25677
WHAT=${1:-game}
mkdir -p outputs outputs/gamengen

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
  game) start_game ;;
  *) echo "用法: $0 [game]   （页面由 nginx 常驻直读，不用起）"; exit 1 ;;
esac
