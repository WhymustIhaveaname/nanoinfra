# Deployment

This document tells you how to start the web playground.
It has two parts: a data preview page and a playable GameNGen demo.

## 1. What runs where

| Part | Port | Bind | Host |
|---|---|---|---|
| nginx, HTTPS, the only public entry | 25676 | 0.0.0.0 | this machine |
| Static files | -- | nginx reads the directory | this machine |
| Inference server, behind `/api/` | 25677 | 127.0.0.1 | 1202b, or this machine |
| Static files, a local spare server | 25675 | 127.0.0.1 | this machine |

The web page is a static site. It needs no GPU. nginx reads the files from the
project directory, so no process of ours has to stay alive for the page to work.

The inference server needs a GPU. It runs the diffusion model.
The page sends one action. The server sends back one frame.

nginx gives one origin to both parts. The page asks `/api/new`, and nginx
passes it to `127.0.0.1:25677/new`.

## 2. Open the page

nginx runs at boot. Nothing to start.

    https://autosr.app:25676/      from outside, a trusted certificate
    https://10.189.12.164:25676/   from the laboratory network, a self-signed
                                   certificate, so accept it once

Plain `http://...:25676/` answers with a redirect to `https`, so an old
bookmark still works.

The data preview tab works now. The play tab needs the inference server.

### Why HTTPS is not optional

From outside the name is `autosr.app`. The whole `.app` suffix is in the HSTS
preload list of Chromium and Edge. Two results, and no server setting changes
either one:

1. `http://autosr.app:25676` is upgraded to `https` by the browser.
2. A self-signed certificate then gives no "proceed anyway" button.

A plain `python -m http.server` receives a TLS ClientHello, answers 400, and
the page does not open at all. The log fills with lines that start `\x16\x03\x01`.
That is a TLS record header. It means the client spoke TLS to a plain server.

The static files and the API must share one origin. A cross-origin `fetch` to
a second port with a self-signed certificate shows no prompt. It fails in
silence. One origin means the reader accepts the certificate once, for both.

### nginx

The site is `/etc/nginx/sites-available/nanoinfra-playground`. Two server
blocks, as the other services on this machine have: `autosr.app` uses the
Let's Encrypt certificate, an IP address uses the self-signed one.

nginx runs as `www-data` and reads the files directly. That works because
`/home/youran` is `751`, the project directories are `775`, and the files are
`664`. nginx also answers Range requests, which `SimpleHTTPRequestHandler`
does not, so the preview videos can seek.

### The spare static server

```bash
cd /home/youran/Nano/nanoinfra/projects/web_playground
bash run.sh page      # 127.0.0.1:25675, local only
```

Use it when nginx is down or when you do not want to touch it. On plain HTTP
the page talks to `http://<host>:25677` directly, as it did before.

## 3. Start the inference server on 1202b

1202b has eight RTX A6000 GPUs. Use 1202b for long sessions.
The local RTX 4060 Ti is 1.67 times slower. Do not hold it for a long time.

```bash
bash run_remote.sh setup    # first time only, about 20 minutes
bash run_remote.sh start    # start the server and the tunnel
bash run_remote.sh status   # show the state
bash run_remote.sh stop     # stop both
```

`setup` does three tasks:

1. It installs `uv`, Python 3.12, and the packages on 1202b.
2. It copies `doom_ngen_server.py`, `models.json`, and `upstream/` to 1202b.
3. It downloads 12 GB of weights from Hugging Face.

`start` does three tasks:

1. It starts the server on 1202b GPU 1.
2. It stops the local inference server, if one runs.
3. It makes an SSH tunnel from this machine to 1202b.

## 4. Start the inference server on this machine

Use this only for a short test. The local GPU is slower.

```bash
bash run.sh game
```

Wait about 20 seconds. The server loads the model and runs a benchmark.
A cached benchmark makes it faster. A first run for a model adds about 8 seconds.

## 5. Important facts

**1202b uses AFS for the home directory. AFS has a 4.8 GB quota.**
The weights are 12 GB. They do not fit in AFS.
Put the weights in `/tmp`. `/tmp` on 1202b is a 2 TB tmpfs.
1202b has 4 TB of RAM. 12 GB of weights plus a 5 GB venv in tmpfs is safe.

**tmpfs loses all files at a reboot.**
Run `setup` again after 1202b reboots.
If only the tunnel stops, run `start`. Do not run `setup` again.

**1202b uses csh as the login shell.**
Do not put shell substitutions in an `ssh host "command"` string.
Pipe a script to `ssh host bash -s` instead.

**An SSH tunnel can attach to a shared master connection. Then it is slow.**
`~/.ssh/config` uses `ControlMaster`. A master connection can stay open for weeks.
A port forward on an old master connection is very slow.
Measured: a ping is 15 ms and the remote loopback is 1.5 ms,
but a forward on a 23-day-old master needed 5 to 9 seconds for each request.
One click then needed more than 10 seconds to make 4 frames.

The forward on a master connection is not a separate process.
`pkill ssh -N` does not stop it. The port stays busy.
A new dedicated tunnel then fails at the bind and stops without a message,
because of `ExitOnForwardFailure`. The health check still measures the old forward.
The fix looks like it works, but nothing changed.

`run_remote.sh` gives the tunnel `-o ControlPath=none`, and `stop` also runs
`ssh -O cancel -L ...` to remove a forward from the master connection.
A dedicated tunnel measures about 50 ms.

**1202b is a shared machine. The server holds the whole GPU.**
The inference server needs only 4.3 GB. An A6000 has 47 GB.
Another user's job takes the free memory and then shares the SM with us.
Our frame rate falls, and `nvidia-smi` only shows a slow card.

The server holds the free memory as ballast, and it manages the ballast itself:

- After a model load, it takes all free memory except `--hold-leave` GiB
  (default 1.5). Measured: it holds 42.5 GiB and leaves 1.0 GiB.
- Before a model load, it gives the ballast back, because the new model
  needs the memory.
- If an inference gets an out-of-memory error, it gives the ballast back and
  tries again. To hold the card must never block our own work.

Verified: four model switches and 30 frames, zero out-of-memory errors.
The ballast is one large tensor. It does not use the SM and it uses no power.

`gpu_hold.py` does the same for the time when the server does not run.
`stop` starts it, so the card stays ours between sessions.
`unhold` gives the card back completely.

**A high load average on 1202b is not a reason for a slow page.**
Measured: load average 44 on 128 cores, and the tunnel was still 52 ms.
The 5-to-9-second delay came from the stale master connection above.
Check the tunnel first. Do not blame the other users.

**Everything but nginx binds to `127.0.0.1`.**
The API has no authentication, so the remote server, the tunnel, and the local
server all bind to the loopback address. nginx is the only public port, and
nginx runs on this machine, so the loopback is enough for it.

Before 2026-09-12 the tunnel bound `0.0.0.0`, because the page asked
`http://<this machine>:25677` from the reader's browser. That put an
unauthenticated GPU endpoint on the public network. The `/api/` route removed
the reason for it.

## 6. Test the deployment

```bash
cd gamengen
.venv/bin/python browsertest.py   # headless Chrome, ~60 checks
.venv/bin/python playtest.py      # scripted play, makes an MP4
```

`browsertest.py` opens the page and makes about 60 checks in eight sections:
page structure, backend, button-to-action, model switch, record and replay,
fault handling, layout, and the data tab.

The page has one input path: a click on an action button. The keyboard and the
mouse-turn control are removed. A click sends four frames of the same action.
Four frames agree with the training data, which holds each action for four frames.

The test clicks the buttons. It does not assign to internal state such as
`G.keys`. Section 4c makes sure that the keyboard does nothing. Keep this rule.
An assignment to internal state hid these three defects, which occurred:

- A key that the model does not support sent 40 requests each second.
- The speed control had no effect, because each mouse move event started a
  second loop.
- The recording did not become empty at a new game.

`playtest.py` sends a fixed action sequence. It saves the frames to a video.
Look at the video. The model must respond to each action.

## 7. Regenerate the data preview

Do this only if you change the data or the sampling.

```bash
cd /home/youran/Nano/nanoinfra
.venv/bin/python projects/web_playground/build_preview.py   # clip comparisons
.venv/bin/python projects/web_playground/build_videos.py    # 60-second videos
```

These scripts read from other directories in the repository:

- `exemplars/nano_world_model/` for `spec`, `encode`, and `codec`
- `datasets/nano_world_model/` for the pixels, 11.2 GB
- `models/video/cosmos_dv4x8x8/` for the codec, 212 MB

## 8. Problems

**The play tab shows "连不上"**
The inference server does not run, or the tunnel is down.
Run `bash run_remote.sh status`.

**The whole page does not open**
Check `systemctl is-active nginx` and `ss -ltnp | grep 25676`.
Then read `/var/log/nginx/error.log`.
A 403 on every file means `www-data` lost its read path; check the modes
listed under nginx above.

**A model load fails**
Read `/tmp/youran-gamengen/server.log` on 1202b.
The three models use different context lengths. Masao uses 64 frames.
arnaudstiegler uses 9 frames. The server reads this from the UNet config.

**All three models stay in the GPU memory.**
One engine needs 4.3 GB, and the card has 47 GB, so the server loads all three
at the start. A model change then only moves a pointer.
Measured: 50 to 93 ms, against 2200 ms for a load with a cached benchmark.
Use `--no-preload-all` for one model only.

The upstream code keeps the context length in a module variable, `BUFFER_SIZE`.
With three models in the memory, one value at load time is not enough:
the value would belong to the model that loaded last.
`Engine.new_session` and `Engine.step` set the value again for their own engine.
Do not remove these two calls. Without them, Masao (64 frames) and
arnaudstiegler (9 frames) mix, and the UNet reports a channel-count error.

**Each frame takes 6 seconds**
A new thread pays a large CUDA initialization cost.
All GPU work must run on one thread. Do not change this.

**The page shows one model, but a different model runs**
This must not occur. The page stops all play when the two do not agree.
If it occurs, reload the page.
