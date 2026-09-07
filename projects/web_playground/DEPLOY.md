# Deployment

This document tells you how to start the web playground.
It has two parts: a data preview page and a playable GameNGen demo.

## 1. What runs where

| Part | Port | Host |
|---|---|---|
| Web page (static files) | 25676 | this machine |
| Inference server | 25677 | 1202b, or this machine |

The web page is a static site. It needs no GPU.

The inference server needs a GPU. It runs the diffusion model.
The page sends one action. The server sends back one frame.

## 2. Start the page

```bash
cd /home/youran/nanoinfra/projects/web_playground
bash run.sh page
```

Open `http://10.189.12.164:25676/` in a browser.

The data preview tab works now. The play tab needs the inference server.

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

**The bind addresses are different on purpose.**
The remote server binds to `127.0.0.1`. The API has no authentication.
Only the SSH tunnel reaches it. Do not open it to the laboratory network.
The tunnel and the local server bind to `0.0.0.0`.
The browser runs on a different machine. It connects to the IP of this machine.

## 6. Test the deployment

```bash
cd gamengen
.venv/bin/python browsertest.py   # headless Chrome, ~60 checks
.venv/bin/python playtest.py      # scripted play, makes an MP4
```

`browsertest.py` opens the page. It sets the input state and calls the step loop.
It checks the button map, the model switch, the load progress, and the tab layout.
It does not dispatch real keyboard or mouse events. The pointer-lock and key
handlers stay untested.

`playtest.py` sends a fixed action sequence. It saves the frames to a video.
Look at the video. The model must respond to each action.

## 7. Regenerate the data preview

Do this only if you change the data or the sampling.

```bash
cd /home/youran/nanoinfra
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

**A model load fails**
Read `/tmp/youran-gamengen/server.log` on 1202b.
The three models use different context lengths. Masao uses 64 frames.
arnaudstiegler uses 9 frames. The server reads this from the UNet config.

**Each frame takes 6 seconds**
A new thread pays a large CUDA initialization cost.
All GPU work must run on one thread. Do not change this.

**The page shows one model, but a different model runs**
This must not occur. The page stops all play when the two do not agree.
If it occurs, reload the page.
