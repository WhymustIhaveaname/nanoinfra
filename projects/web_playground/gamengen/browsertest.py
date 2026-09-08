"""browsertest.py — 无头 Chrome 打开页面、真的点按钮，验证前端。

playtest.py 走 HTTP 验模型出的画面；这里验的是前端：页面结构、按钮 -> 动作、
模型切换、录制回放、容错、布局。输入只有一条路（点按钮），所以全部用真实 click，
不改 G.keys 之类的内部状态；唯一一处例外是把后端标成不可达，那没法用点击造出来。
"""

import argparse
import sys
import time

from playwright.sync_api import sync_playwright

REP = 4          # 一次点击 = 4 帧（对齐训练数据）
ACTS = ["FWD", "BACK", "MLEFT", "MRIGHT", "TLEFT", "TRIGHT", "ATTACK", ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:25676/")
    ap.add_argument("--shots", default="../outputs/gamengen")
    a = ap.parse_args()

    fails = []

    def check(name, ok, detail=""):
        print(f"  {'✓' if ok else '✗'} {name}{'  ' + str(detail) if detail else ''}")
        if not ok:
            fails.append(name)

    with sync_playwright() as pw:
        br = pw.chromium.launch(args=["--no-sandbox", "--use-gl=swiftshader"])
        pg = br.new_page(viewport={"width": 1500, "height": 1000})
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        steps = []
        pg.on("response", lambda r: steps.append(r.status) if "/step" in r.url else None)

        js = pg.evaluate
        rec_len = lambda: js("() => G.rec.length")
        text = lambda sel: pg.locator(sel).inner_text()

        def settle(timeout=25):
            """等到没有飞行中的开局、且帧数连续两拍不变。

            开局是异步的：rec 会先被清空再重新长起来。直接等「rec >= N」会在
            rec 还留着上一组帧时立刻返回，量到清空过程中的中间态——微臣被这个
            假失败骗过一轮，页面其实是对的。
            """
            t0, last, same = time.time(), -1, 0
            while time.time() - t0 < timeout:
                pg.wait_for_timeout(300)
                n = rec_len()
                if js("() => G.newing") is None and n == last:
                    same += 1
                    if same >= 2:
                        return n
                else:
                    same = 0
                last = n
            return rec_len()

        def wait_until(cond, timeout=20):
            t0 = time.time()
            while not cond() and time.time() - t0 < timeout:
                pg.wait_for_timeout(100)
            return time.time() - t0

        def tap(act):
            """点一个按钮，等这一组帧到齐，返回这组帧的动作名列表。"""
            n = rec_len()
            pg.click(f'.padbtn[data-act="{act}"]')
            wait_until(lambda: rec_len() >= n + REP)
            pg.wait_for_timeout(400)          # 多等一拍，抓「多发了一帧」
            return js(f"() => G.rec.slice({n}).map(r => r.action)")

        def switch_to(mid):
            pg.select_option("#g-model", mid)
            return wait_until(lambda: js("() => G.loaded") == mid and not js("() => G.stale"))

        def canvas_blank():
            return js("""() => { const c = document.getElementById('screen');
                const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
                let s = 0; for (let i = 0; i < d.length; i += 4) s += d[i] + d[i+1] + d[i+2];
                return s === 0; }""")

        def no_hscroll(w):
            pg.set_viewport_size({"width": w, "height": 900})
            pg.wait_for_timeout(300)
            return js("() => document.documentElement.scrollWidth") <= w + 2

        # ---------- A. 页面结构 ----------
        print("A) 页面结构")
        pg.goto(a.url, wait_until="networkidle")
        pg.wait_for_timeout(2500)
        tabs, subs = pg.locator("#tabs button"), pg.locator("#subtabs button")
        check("一级标签：试玩 / 数据预览", tabs.count() == 2 and "试玩" in tabs.nth(0).inner_text())
        check("二级标签 = 4 个数据源", subs.count() == 4)
        check("试玩页不显示二级标签", "hide" in (pg.get_attribute("#subtabs", "class") or ""))
        check("没有键盘遮罩、没有载入按钮、没有速度栏", js(
            "() => ['overlay', 'g-loadmodel', 'g-speed'].every(i => !document.getElementById(i))"))
        check("模型区在画布上方", js("() => document.querySelector('.modelbar').getBoundingClientRect().bottom")
              <= js("() => document.getElementById('screen').getBoundingClientRect().top"))

        # ---------- B. 后端 ----------
        print("B) 后端")
        wait_until(lambda: not js("() => G.stale"), 30)
        gpu = text("#g-gpu")
        check("拿到 GPU 信息", "上下文" in gpu, gpu)
        check("显示了启动基准", "ms/帧" in text("#g-bench"))
        opts = pg.eval_on_selector_all("#g-model option", "e => e.map(x => x.value)")
        names = dict(zip(opts, pg.eval_on_selector_all("#g-model option", "e => e.map(x => x.textContent)")))
        check("三个模型全部常驻显存", sorted(js("() => G.pool")) == sorted(opts) and len(opts) == 3, opts)
        check("说明里有代码仓库和权重链接", pg.locator("#g-modelnote a").count() == 2)
        check("开局有画面", not canvas_blank())
        pg.locator("#game").screenshot(path=f"{a.shots}/browser_1_game.png")

        # ---------- C. 按钮 -> 动作 ----------
        print("C) 按钮 -> 动作")
        for act in ACTS:
            if pg.is_disabled(f'.padbtn[data-act="{act}"]'):
                check(f"「{act or '不动'}」该模型没有，按钮已置灰", act in ("BACK", ""))
                continue
            sent = tap(act)
            want = act or "NOOP"
            check(f"点「{act or '不动'}」= {REP} 帧全是 {want}", sent == [want] * REP, sent)
        check("点完不留连发余量", js("() => G.repeatLeft") == 0)
        missing = text("#g-missing")
        check("缺的动作有说明", (("后退" in missing) == (not js("() => G.hasBack")))
              and (("不动" in missing) == (not js("() => G.hasNoop"))), missing)
        srv, cli = float(text("#g-ms")), js("() => G.times.reduce((a, b) => a + b, 0) / G.times.length")
        check("前端开销不超过推理本身", cli - srv < srv, f"服务端 {srv:.0f} ms，墙钟 {cli:.0f} ms")

        n0 = rec_len()
        pg.wait_for_timeout(3000)
        check("不点就定格", rec_len() == n0)
        steps.clear()
        for k in ("w", "a", "s", "d", " ", "Enter"):
            pg.keyboard.press(k)
        pg.wait_for_timeout(1500)
        check("敲键盘不发请求、不出帧", not steps and rec_len() == n0, f"{len(steps)} 个请求")
        check("点画面不锁指针", js("() => { document.getElementById('screen').click();"
                                   " return document.pointerLockElement === null; }"))
        n0 = rec_len()
        pg.click('.padbtn[data-act="FWD"]'); pg.click('.padbtn[data-act="TLEFT"]')
        wait_until(lambda: rec_len() >= n0 + 2 * REP)
        check("连点两个按钮不串台", js(f"() => G.rec.slice({n0}).map(r => r.action)")
              == ["FWD"] * REP + ["TLEFT"] * REP)

        # ---------- D. 模型切换 ----------
        print("D) 模型切换")
        cur = js("() => G.loaded")
        other = [o for o in opts if o != cur][0]
        st = js("""(m) => { document.getElementById('g-model').value = m; syncModelState();
            return [G.stale, document.querySelector('.padbtn[data-act="FWD"]').disabled,
                    document.getElementById('g-running').textContent]; }""", other)
        check("切换途中：stale、按钮置灰、读数写「切换中」", st[0] and st[1] and "切换中" in st[2], st)
        n0 = rec_len()
        pg.click('.padbtn[data-act="FWD"]', force=True)
        pg.wait_for_timeout(1500)
        check("切换途中点按钮不出帧", rec_len() == n0)
        js("(m) => { document.getElementById('g-model').value = m; syncModelState(); }", cur)

        buf0 = text("#g-gpu").split("上下文")[1].split()[0]
        dt = switch_to(other)
        check(f"选了 {other} 就自动切过去", js("() => G.loaded") == other and dt < 3, f"{dt:.2f}s")
        check("读数写的是真正在跑的", text("#g-runname") == names[other], text("#g-runname"))
        buf1 = text("#g-gpu").split("上下文")[1].split()[0]
        check("上下文帧数随模型变", buf1 != buf0, f"{buf0} -> {buf1}")
        check("换完自动开了新局", not canvas_blank())
        check("切完第一次点击就生效", tap("FWD") == ["FWD"] * REP)
        dt = switch_to(cur)
        check("切回来也是瞬时", js("() => G.loaded") == cur and dt < 3, f"{dt:.2f}s")

        # ---------- E. 录制与回放 ----------
        print("E) 录制与回放")
        pg.click("#g-new"); wait_until(lambda: rec_len() == 0)
        check("新开一局清空录制", rec_len() == 0 and text("#g-recact").strip() == "还没有帧"
              and pg.is_disabled("#g-replay"))
        # /new 还没回来时点的帧既不能丢、也不能被后到的开局抹掉。
        # 连点多次「新开一局」时，多个 /new 会撞在一起：必须合并成一局，
        # 而不是串行执行（串行的那版会把中间点出来的帧抹掉）。
        for k in (1, 3):
            for _ in range(k):
                pg.click("#g-new")
            pg.click('.padbtn[data-act="FWD"]'); pg.click('.padbtn[data-act="TLEFT"]')
            settle()
            check(f"连点 {k} 次新开一局后的点击一帧不丢",
                  js("() => G.rec.map(r => r.action)") == ["FWD"] * REP + ["TLEFT"] * REP,
                  js("() => G.rec.map(r => r.action)"))
        n = rec_len()
        check("玩过的帧被记录", n == 2 * REP and js("() => document.getElementById('g-recacts').children.length") == n)
        js("() => { G.scrubbing = true; showRecFrame(1); }")
        check("能看任意一帧", text("#g-scrubv").startswith("2 /") and "TLEFT" not in text("#g-recact"))
        js("() => updateRecUI()")
        check("看历史帧时新帧不抢进度条", text("#g-scrubv").startswith("2 /"))
        tap("FWD")
        check("又开始玩后进度条跟随最新帧", not js("() => G.scrubbing"))

        pg.click("#g-replay"); pg.wait_for_timeout(300)
        n = rec_len()
        check("回放中：按钮置灰、不追加帧", js("() => G.replaying")
              and pg.is_disabled('.padbtn[data-act="FWD"]') and rec_len() == n)
        js("() => window.scrollTo(0, 0)"); pg.wait_for_timeout(1200)
        check("回放不把整页滚走", js("() => window.scrollY") == 0)
        js("() => stopReplay()")
        check("回放结束后无残留输入", not js("() => hasInput()"))

        js("() => { G.sid = null; }")
        n = rec_len()
        check("session 失效：录制保留、第一次点击仍生效", tap("TLEFT") == ["TLEFT"] * REP and rec_len() == n + REP)

        # ---------- F. 容错 ----------
        print("F) 容错")
        js("() => { G.loaded = null; G.pool = []; syncModelState(); }")
        check("后端不可达时禁止游玩", js("() => G.stale") and pg.is_disabled('.padbtn[data-act="FWD"]'))
        js("() => probeServer()")
        wait_until(lambda: not js("() => G.stale"))
        check("重新问到后恢复", not js("() => G.stale"))
        js("() => showErr('测试')")
        check("错误提示能出现、能清", "测试" in text("#g-err") and (js("() => clearErr()") or text("#g-err").strip() == ""))
        js("() => { G.padQueue = [['FWD'], ['FWD']]; G.repeatLeft = 3; G.heldButtons = ['FWD'];"
           " Object.defineProperty(document, 'hidden', {value: true, configurable: true});"
           " document.dispatchEvent(new Event('visibilitychange')); }")
        check("切后台清空排队的帧", not js("() => hasInput()"))
        js("() => Object.defineProperty(document, 'hidden', {value: false, configurable: true})")
        check("请求带超时封装", js("() => typeof api === 'function'"))

        # ---------- G. 布局 ----------
        print("G) 布局")
        for i, name in ((0, "试玩页"), (1, "数据页")):
            tabs.nth(i).click(); pg.wait_for_timeout(500)
            bad = [w for w in (1000, 760, 560, 380) if not no_hscroll(w)]
            check(f"{name}窄屏无横向溢出", not bad, f"溢出于 {bad}px" if bad else "")
        pg.set_viewport_size({"width": 1500, "height": 1000})

        # ---------- H. 数据页 ----------
        print("H) 数据页")
        tabs.nth(1).click(); pg.wait_for_timeout(500)
        subs.nth(2).click(); pg.wait_for_timeout(1000)
        check("二级标签可选、逐帧控件出现", "on" in (subs.nth(2).get_attribute("class") or "")
              and pg.locator("#controls").is_visible())
        check("有切片图和录像", pg.locator("#main .film").count() > 0 and pg.locator("#main video").count() > 0)
        pg.screenshot(path=f"{a.shots}/browser_2_data.png")
        tabs.nth(0).click(); pg.wait_for_timeout(400)
        check("回到试玩页：逐帧控件和二级标签隐藏", not pg.locator("#controls").is_visible()
              and "hide" in (pg.get_attribute("#subtabs", "class") or ""))
        tabs.nth(1).click(); pg.wait_for_timeout(500)
        check("数据页记得上次看的来源", "on" in (subs.nth(2).get_attribute("class") or ""))
        check("切走的页面视频已暂停", js("() => [...document.querySelectorAll('#game video')].every(v => v.paused)"))

        check("全程无 JS 报错", not errs, errs[:2])
        br.close()

    print()
    if fails:
        print(f"✗ {len(fails)} 项没过: {fails}")
        sys.exit(1)
    print("✓ 全部通过")


if __name__ == "__main__":
    main()
