#!/usr/bin/env bash
# run_remote.sh — 把推理搬到 1202b 的一张 A6000 上，本机只留网页。
#
# 为什么：本机 4060 Ti 不能长期占用，而 1202b 有 8 张 A6000 48GB。
# 顺带还快 1.67 倍——A6000 上 182.7 ms/帧，4060 Ti 上 304.5 ms/帧。
#
# 怎么连：浏览器多半够不着校内的 1202b，所以用 SSH 端口转发把远端的 25677
# 映射到本机同一个端口。页面里那个「推理服务」地址一个字都不用改，
# 它看到的还是 <本机>:25677，只是背后换成了 A6000。
#
# 两侧绑定故意不一样：远端只听 127.0.0.1（这接口没有认证，不该对整个实验室
# 网络敞开，只有 SSH 隧道进得来就够了）；本机隧道听 0.0.0.0，因为页面是在
# 陛下的浏览器里跑的，它连的是这台机器的 IP 而不是 127.0.0.1。
#
# 落在 1202b 的哪里：/tmp。那台机器 /tmp 是 2TB tmpfs 而它有 4TB 内存，
# 放 12GB 权重加 5GB venv 毫无压力；AFS 家目录只有 4.8GB 配额装不下，而 /var/tmp 只剩 23G
# 且属于系统 /var，写满会伤到机器。代价是重启后要重跑本脚本的 setup。
#
#   ./run_remote.sh setup     # 首次：装环境 + 下权重（约 20 分钟）
#   ./run_remote.sh start     # 起远端服务 + 本机隧道
#   ./run_remote.sh stop      # 停隧道和远端服务
#   ./run_remote.sh status
#
# 换回本机跑：./run_remote.sh stop 然后 ./run.sh game

set -u
cd "$(dirname "$0")"
HOST=${GAMENGEN_HOST:-1202b}
GPU=${GAMENGEN_GPU:-1}          # 远端用第几张卡
PORT=25677
B=/tmp/youran-gamengen

mkdir -p outputs outputs/gamengen

remote() { ssh -o BatchMode=yes "$HOST" bash -s; }

case "${1:-status}" in

setup)
  echo "[1/3] 装 uv + venv + 依赖 -> $HOST:$B"
  remote <<EOS
set -e
mkdir -p $B; cd $B
[ -x $B/uv ] || curl -LsSf https://astral.sh/uv/install.sh | \
    env UV_INSTALL_DIR=$B INSTALLER_NO_MODIFY_PATH=1 sh >/dev/null
$B/uv venv --python 3.12 $B/.venv 2>&1 | tail -1
# torchvision / datasets / peft 是上游 dataset.py 和 model.py 的间接依赖，
# 少一个 doom_ngen_server.py 就 import 不进来。
VIRTUAL_ENV=$B/.venv $B/uv pip install --no-cache \
  "numpy<2.0.0" torch torchvision diffusers transformers accelerate \
  safetensors pillow huggingface_hub datasets peft 2>&1 | tail -1
$B/.venv/bin/python -c "import torch; print('torch', torch.__version__, '| 卡数', torch.cuda.device_count())"
EOS
  echo "[2/3] 传代码"
  tar cz gamengen/doom_ngen_server.py gamengen/models.json gamengen/upstream \
    | ssh -o BatchMode=yes "$HOST" "mkdir -p $B/app && tar xz -C $B/app"
  echo "[3/3] 下权重（约 12GB）"
  remote <<EOS
$B/.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download, HfApi
base = "$B/weights"
for repo, sub in [("Masao-Taketani/vizdoom-diffusion-dynamic-model", "unet"),
                  ("Masao-Taketani/vizdoom-finetuned-decoder", "vae"),
                  ("arnaudstiegler/sd-model-gameNgen-60ksteps", "arnaud60k"),
                  ("arnaudstiegler/gameNgen-baseline-20ksteps", "arnaud20k")]:
    snapshot_download(repo, local_dir=f"{base}/{sub}"); print("ok", sub, flush=True)
i = HfApi().repo_info("Masao-Taketani/vizdoom-inference-latent-dataset",
                      repo_type="dataset", files_metadata=True)
pts = sorted([f for f in i.siblings if f.rfilename.endswith(".pt")], key=lambda x: x.size or 0)
hf_hub_download("Masao-Taketani/vizdoom-inference-latent-dataset",
                pts[len(pts)//4].rfilename, repo_type="dataset", local_dir=f"{base}/latents")
print("ok latents")
PY
du -sh $B/weights
EOS
  echo "setup 完成，接着 ./run_remote.sh start"
  ;;

start)
  echo "起远端服务（$HOST GPU $GPU）"
  remote <<EOS
pkill -u \$(whoami) -f "doom_ngen_server.py --port $PORT" 2>/dev/null
sleep 2
cd $B/app/gamengen
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU \
  nohup $B/.venv/bin/python doom_ngen_server.py --port $PORT \
        --bind 127.0.0.1 --bench 25 --base $B/weights \
        > $B/server.log 2>&1 &
echo "  远端 PID \$!"
EOS
  echo "  等模型载入…"
  for i in $(seq 1 40); do
    sleep 5
    if ssh -o BatchMode=yes "$HOST" "grep -q '\[serve\]' $B/server.log" 2>/dev/null; then break; fi
  done
  ssh -o BatchMode=yes "$HOST" "grep -E 'bench|serve' $B/server.log | tail -2"

  # 本机若有服务占着这个端口，先让位
  LOCAL=$(ps -u "$(whoami)" -o pid,cmd | grep "[d]oom_ngen_server.py --port $PORT" | awk '{print $1}')
  # 不加引号：$LOCAL 可能是多个 pid，引起来会被当成单个参数而 kill 失败
  [ -n "$LOCAL" ] && kill $LOCAL && echo "  本机推理服务已停，显卡释放"
  pkill -u "$(whoami)" -f "ssh -N .*:$PORT:127.0.0.1:$PORT" 2>/dev/null
  sleep 2
  nohup ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 -L "0.0.0.0:$PORT:127.0.0.1:$PORT" "$HOST" \
        > outputs/gamengen/tunnel.log 2>&1 &
  echo "  隧道 PID $!"
  sleep 5
  curl -s -m 20 "http://127.0.0.1:$PORT/info" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print('  通了:', d['gpu'], '|', d['bench']['mean_ms'], 'ms/帧')" \
    || { echo "  隧道没通，看 outputs/gamengen/tunnel.log"; exit 1; }
  # 隧道健康检查。一条久开的隧道会劣化：实测新隧道空转 41ms，而连开数小时、
  # 被反复 kill 重连过的老隧道要 830ms——足以把 4.9 fps 拖成 1 fps，
  # 而且症状看起来像「远端 GPU 慢」，极易误判。所以每次 start 都量一下。
  echo -n "  隧道空转延迟: "
  python3 - <<'PYEOF'
import http.client, time
c = http.client.HTTPConnection('127.0.0.1', 25677, timeout=30)
ts = []
for _ in range(6):
    t0 = time.time(); c.request('GET', '/status'); r = c.getresponse(); r.read()
    ts.append((time.time() - t0) * 1000)
ms = sorted(ts)[len(ts) // 2]
print(f"{ms:.0f} ms" + ("" if ms < 150 else "  ← 偏高！隧道可能已劣化，重跑 stop 再 start"))
PYEOF
  ;;

stop)
  pkill -u "$(whoami)" -f "ssh -N .*:$PORT:127.0.0.1:$PORT" && echo "隧道已停"
  remote <<EOS
pkill -u \$(whoami) -f "doom_ngen_server.py --port $PORT" && echo "远端服务已停" || echo "远端本来就没跑"
EOS
  ;;

status)
  echo "本机隧道: $(pgrep -u "$(whoami)" -f "ssh -N .*:$PORT:127.0.0.1:$PORT" >/dev/null && echo 在 || echo 无)"
  curl -s -m 10 "http://127.0.0.1:$PORT/info" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print('  端点:', d['gpu'], '| 载入', d['loaded'])" \
    2>/dev/null || echo "  :$PORT 无响应"
  remote <<EOS
echo "远端进程: \$(pgrep -u \$(whoami) -f 'doom_ngen_server.py --port $PORT' >/dev/null && echo 在 || echo 无)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed -n "\$((${GPU}+1))p"
EOS
  ;;

*) echo "用法: $0 [setup|start|stop|status]"; exit 1 ;;
esac
