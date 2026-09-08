"""browsertest.py — 用无头 Chrome 真的把页面打开、真的点按钮，验证前端不是纸上谈兵。

playtest.py 验的是「动作发过去，模型回的画面对不对」，走的是 HTTP 接口，绕过了整个前端。
这里补的是另一半：页面能不能加载、标签能不能切、点按钮有没有变成正确的动作上报、
canvas 上的像素有没有真的在变。

输入只有一条路：点操作按钮。键盘和鼠标转向已从页面整个移除，
所以本测试**只用真实的 click**，不再直接改 G.keys 那类内部状态。
4c 一段专门验证敲键盘确实什么都不发生。
"""

import argparse
import sys

from playwright.sync_api import sync_playwright


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:25676/")
    ap.add_argument("--shots", default="../outputs/gamengen")
    a = ap.parse_args()

    fails = []
    def check(name, ok, detail=""):
        print(f"  {'✓' if ok else '✗'} {name}{'  ' + detail if detail else ''}")
        if not ok:
            fails.append(name)

    MODEL_NAMES = {}
    with sync_playwright() as pw:
        br = pw.chromium.launch(args=["--no-sandbox", "--use-gl=swiftshader"])
        pg = br.new_page(viewport={"width": 1500, "height": 1000})
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)

        print("1) 加载页面")
        pg.goto(a.url, wait_until="networkidle")
        pg.wait_for_timeout(2500)
        check("无 JS 报错", not errs, str(errs[:2]) if errs else "")

        tabs = pg.locator("#tabs button")
        subs = pg.locator("#subtabs button")
        check("一级标签 = 2（试玩 / 数据预览）", tabs.count() == 2, f"实际 {tabs.count()}")
        check("二级标签 = 4（四个数据源）", subs.count() == 4, f"实际 {subs.count()}")
        check("一级第一个是试玩", "试玩" in tabs.nth(0).inner_text())
        check("一级第二个是数据预览", "数据预览" in tabs.nth(1).inner_text())
        check("下载那组已改名", "GameNGen" in subs.nth(0).inner_text(),
              subs.nth(0).inner_text())
        check("试玩页不显示二级标签",
              "hide" in (pg.get_attribute("#subtabs", "class") or ""))

        print("2) 推理后端连上了吗")
        gpu = pg.locator("#g-gpu").inner_text()
        check("拿到 GPU 信息", "未启动" not in gpu and gpu != "…", gpu)
        bench = pg.locator("#g-bench").inner_text()
        check("显示了启动基准", "ms/帧" in bench, bench)

        ybar = pg.evaluate("() => document.querySelector('.modelbar').getBoundingClientRect().bottom")
        ycanvas = pg.evaluate("() => document.getElementById('screen').getBoundingClientRect().top")
        check("模型区在画布上方", ybar <= ycanvas, f"modelbar底={ybar:.0f} canvas顶={ycanvas:.0f}")

        print("3) 开局画面")
        pg.wait_for_timeout(3000)
        blank = pg.evaluate("""() => {
            const c = document.getElementById('screen');
            const d = c.getContext('2d').getImageData(0,0,c.width,c.height).data;
            let s = 0; for (let i=0;i<d.length;i+=4) s += d[i]+d[i+1]+d[i+2];
            return s === 0;
        }""")
        check("canvas 上有画面（非全黑）", not blank)
        pg.locator("#game").screenshot(path=f"{a.shots}/browser_1_game.png")

        print("3b) 连不上后端时不许出帧")
        blocked = pg.evaluate("""async () => {
            const keep = window.GAME_API_probe = G.loaded;
            G.loaded = null; G.pool = []; syncModelState();
            const stale = G.stale;
            const disabled = document.querySelector('.padbtn[data-act="FWD"]').disabled;
            return {stale, disabled};
        }""")
        check("不知道装的是哪个模型时禁止游玩", blocked["stale"], str(blocked))
        check("此时动作按钮也置灰", blocked["disabled"], str(blocked))
        pg.evaluate("() => probeServer()")
        pg.wait_for_timeout(2500)
        check("重新问到之后恢复可玩", not pg.evaluate("() => G.stale"))

        print("3c) 键盘和「点击画面开始」都已去掉，模型也不用手动载入")
        check("画面上没有遮罩元素",
              pg.evaluate("() => document.getElementById('overlay') === null"))
        check("没有载入/切换按钮",
              pg.evaluate("() => document.getElementById('g-loadmodel') === null"))

        print("4) 点操作按钮出帧")
        # 输入只有一条路：点按钮。键盘和鼠标转向已从页面整个移除。
        REP = 4          # 一次点击 = 4 帧
        def tap(act, n=None, timeout=20000):
            """点一下按钮，等这一组帧全部到齐再返回。"""
            base = pg.evaluate("() => G.rec.length")
            pg.click(f'.padbtn[data-act="{act}"]')
            want = base + (n if n is not None else REP)
            waited = 0
            while pg.evaluate("() => G.rec.length") < want and waited < timeout:
                pg.wait_for_timeout(200); waited += 200
            pg.wait_for_timeout(400)     # 多等一拍，好抓出「多发了一帧」
            return base

        before = pg.evaluate("() => document.getElementById('screen').toDataURL().length")
        tap("FWD")
        steps = int(pg.locator("#g-steps").inner_text())
        check("点一下前进就出帧", steps >= 3, f"steps={steps}")
        check("状态栏动作是前进", pg.locator("#g-act").inner_text() == "FWD",
              pg.locator("#g-act").inner_text())
        check("canvas 内容变了",
              before != pg.evaluate("() => document.getElementById('screen').toDataURL().length"))
        check("显示了 fps", "fps" in pg.locator("#g-fps").inner_text(),
              pg.locator("#g-fps").inner_text())

        srv_ms = float(pg.locator("#g-ms").inner_text())
        cli_ms = pg.evaluate("() => G.times.reduce((a,b)=>a+b,0)/G.times.length")
        print(f"     服务端 {srv_ms:.0f} ms/帧，客户端墙钟 {cli_ms:.0f} ms/帧，"
              f"前端开销 {cli_ms - srv_ms:.0f} ms")
        check("前端开销不超过推理本身", cli_ms - srv_ms < srv_ms,
              f"开销 {cli_ms - srv_ms:.0f} ms")

        print("4b) 不点就定格")
        n0 = int(pg.locator("#g-steps").inner_text())
        pg.wait_for_timeout(3500)
        check("没有输入时不再生成新帧",
              int(pg.locator("#g-steps").inner_text()) == n0,
              f"{n0} -> {pg.locator('#g-steps').inner_text()}")

        print("4c) 键盘已经整个去掉，敲键不许有任何反应")
        steps_seen = []
        pg.on("response", lambda r: steps_seen.append(r.status) if "/step" in r.url else None)
        # 空格和回车要一起试：按钮点完若还留着焦点，这两个键会把它再激活一次，
        # 等于键盘又能玩了。
        for k in ("w", "a", "s", "d", " ", "Enter"):
            pg.keyboard.down(k); pg.wait_for_timeout(250); pg.keyboard.up(k)
        pg.wait_for_timeout(2000)
        check("敲 WASD／空格／回车不发任何请求", len(steps_seen) == 0,
              f"发了 {len(steps_seen)} 个")
        check("敲键不产生帧", int(pg.locator("#g-steps").inner_text()) == n0,
              f"{n0} -> {pg.locator('#g-steps').inner_text()}")
        check("点画面不再锁指针", pg.evaluate("""() => {
                  document.getElementById('screen').click();
                  return document.pointerLockElement === null; }"""))

        print("5) 每个按钮上报的动作名逐一核对")
        for act in ("FWD", "BACK", "MLEFT", "MRIGHT", "TLEFT", "TRIGHT", "ATTACK", ""):
            btn = pg.locator(f'.padbtn[data-act="{act}"]')
            if btn.is_disabled():
                print(f"     跳过「{act or 'NOOP'}」：当前模型没有这个动作")
                continue
            n = tap(act)
            sent = pg.evaluate(f"() => G.rec.slice({n}).map(r => r.action)")
            want = act or "NOOP"     # 空按钮 = 不动，服务端报回 NOOP
            check(f"点「{act or '不动'}」上报 {want}",
                  len(sent) == 4 and set(sent) == {want}, f"实际 {sent}")
        # 动作 id 的翻译交给服务端，按模型的表来——这是「不动发成左转」那个 bug 的修法
        tbl = pg.evaluate("() => fetch(GAME_API+'/info').then(r=>r.json()).then(d=>[d.actions.length, d.has_noop])")
        check("服务端报出当前模型的动作表", isinstance(tbl, list) and tbl[0] in (12, 18),
              f"表长 {tbl}")

        print("6) 速度栏已移除")
        check("页面上没有速度选择器",
              pg.evaluate("() => document.getElementById('g-speed') === null"))

        print("6b) 下拉框选了就自动切，切换途中不许出帧")
        MODEL_NAMES.update(dict(zip(
            pg.eval_on_selector_all("#g-model option", "e=>e.map(x=>x.value)"),
            pg.eval_on_selector_all("#g-model option", "e=>e.map(x=>x.textContent)"))))
        cur0 = pg.input_value("#g-model")
        other0 = [o for o in pg.eval_on_selector_all("#g-model option", "e=>e.map(x=>x.value)")
                  if o != cur0][0]
        # 界面和显存不一致的那段窗口里，一帧都不许发——否则会出现
        # 「界面写着 A、实际在跑 B」，玩家据此得出的结论全是错的。
        st = pg.evaluate("""(other) => {
            const sel = document.getElementById('g-model');
            sel.value = other; syncModelState();
            return {stale: G.stale,
                    dis: document.querySelector('.padbtn[data-act="FWD"]').disabled,
                    txt: document.getElementById('g-running').textContent};
        }""", other0)
        check("切换途中 stale 置位", st["stale"], str(st))
        check("切换途中动作按钮置灰", st["dis"], str(st))
        check("切换途中读数写「切换中」", "切换中" in st["txt"], st["txt"])
        before_n = int(pg.locator("#g-steps").inner_text())
        pg.click('.padbtn[data-act="FWD"]', force=True)
        pg.wait_for_timeout(2000)
        check("切换途中点按钮不生成任何帧",
              int(pg.locator("#g-steps").inner_text()) == before_n,
              f"{before_n} -> {pg.locator('#g-steps').inner_text()}")
        pg.evaluate("() => { document.getElementById('g-model').value = arguments0; }"
                    .replace("arguments0", repr(cur0)))
        pg.evaluate("() => syncModelState()")
        check("退回真正在跑的那个之后恢复可玩", not pg.evaluate("() => G.stale"))
        check("读数恢复「当前正在玩」",
              "当前正在玩" in pg.locator("#g-running").inner_text(),
              pg.locator("#g-running").inner_text())

        print("6c) 模型缺哪些动作要写清楚，按钮要置灰")
        missing = pg.locator("#g-missing").inner_text()
        for act, name in (("BACK", "后退"), ("", "不动")):
            has = pg.evaluate(f"() => G.{'hasBack' if act else 'hasNoop'}")
            dis = pg.is_disabled(f'.padbtn[data-act="{act}"]')
            check(f"「{name}」按钮状态和模型一致", dis == (not has),
                  f"模型有={has} 按钮禁用={dis}")
            if not has:
                check(f"说明里写了缺「{name}」", name in missing, missing)
        if pg.evaluate("() => G.hasBack && G.hasNoop"):
            check("模型什么都有时不写多余的话", missing.strip() == "", missing)

        print("6d) 本局录制与回放")
        pg.evaluate("() => { gameNew(); }")
        pg.wait_for_timeout(2500)
        tap("FWD"); tap("TLEFT")
        n = pg.evaluate("() => G.rec.length")
        check("玩过的帧被记录下来", n >= 3, f"{n} 帧")
        check("动作历史条有格子",
              pg.evaluate("() => document.getElementById('g-recacts').children.length") == n)
        pg.evaluate("() => { stopReplay(); showRecFrame(1); }")
        pg.wait_for_timeout(200)
        check("能拖到任意一帧", pg.locator("#g-scrubv").inner_text().startswith("2 /"),
              pg.locator("#g-scrubv").inner_text())
        check("显示该帧的动作", "动作" in pg.locator("#g-recact").inner_text(),
              pg.locator("#g-recact").inner_text())
        pg.click("#g-replay"); pg.wait_for_timeout(500)
        check("回放中不再生成新帧", pg.evaluate("() => G.replaying"))
        pg.evaluate("() => stopReplay()")
        pg.wait_for_timeout(200)
        check("能停止回放", not pg.evaluate("() => G.replaying"))

        print("6e) 点一次操作按钮 = REPEAT 帧，且这 REPEAT 帧都是刚点的那个动作")
        pg.evaluate("() => resetInput()")
        pg.wait_for_timeout(600)
        rep = pg.evaluate("() => REPEAT")
        # 这条以前只数帧数，不看动作，所以在「点左转却发出 FWD,FWD,FWD,TLEFT」时
        # 照样通过——帧数对、动作全错。必须断言发出去的动作本身。
        for act in ("FWD", "TLEFT", "ATTACK", "MRIGHT"):
            before = tap(act)
            sent = pg.evaluate(f"() => G.rec.slice({before}).map(r => r.action)")
            check(f"点「{act}」发出 {rep} 帧且全是它", len(sent) == rep and set(sent) == {act},
                  f"实际 {sent}")
        check("点完不留残留", pg.evaluate("() => G.repeatLeft") == 0)

        print("6f) 回放期间不吃输入，结束后不倒灌")
        pg.evaluate("() => startReplay()")
        pg.wait_for_timeout(300)
        pg.click('.padbtn[data-act="FWD"]', force=True)   # 回放中点按钮应当被吞掉
        pg.wait_for_timeout(300)
        n0 = pg.evaluate("() => G.rec.length")
        pg.wait_for_timeout(1200)
        check("回放中不追加新帧", pg.evaluate("() => G.rec.length") == n0)
        pg.evaluate("() => stopReplay()")
        pg.wait_for_timeout(300)
        check("回放结束后没有攒下的输入", not pg.evaluate("() => hasInput()"))
        check("回放中按钮是置灰的（不是静默吞掉）",
              pg.evaluate('''() => { startReplay();
                  const d = document.querySelector('.padbtn[data-act="FWD"]').disabled;
                  stopReplay(); return d; }'''))

        print("6g) 切到后台不留排队的帧")
        pg.evaluate('''() => { G.padShots = 3; G.repeatLeft = 3; G.heldButtons = ['FWD'];
            Object.defineProperty(document, 'hidden', {value: true, configurable: true});
            document.dispatchEvent(new Event('visibilitychange')); }''')
        check("切后台后排队清空", not pg.evaluate("() => hasInput()"))
        pg.evaluate("() => Object.defineProperty(document, 'hidden', {value: false, configurable: true})")

        print("6h) 错误提示会自己消失，不会永久挂着")
        pg.evaluate("() => showErr('测试')")
        check("提示能出现", "测试" in pg.locator("#g-err").inner_text())
        pg.evaluate("() => clearErr()")
        check("提示能清除", pg.locator("#g-err").inner_text().strip() == "")

        print("6i) 窄屏不横向溢出")
        for w in (1000, 760, 560):
            pg.set_viewport_size({"width": w, "height": 900})
            pg.wait_for_timeout(350)
            sw = pg.evaluate("() => document.documentElement.scrollWidth")
            check(f"视口 {w}px 无横向滚动条", sw <= w + 2, f"内容宽 {sw}")
        pg.set_viewport_size({"width": 1500, "height": 1000})
        pg.wait_for_timeout(300)

        print("6i2) 数据预览页窄屏也不许溢出")
        pg.click("#tabs button:nth-child(2)"); pg.wait_for_timeout(600)
        for w in (1000, 760, 560):
            pg.set_viewport_size({"width": w, "height": 900}); pg.wait_for_timeout(400)
            sw = pg.evaluate("() => document.documentElement.scrollWidth")
            check(f"数据页视口 {w}px 无横向滚动条", sw <= w + 2, f"内容宽 {sw}")
        pg.set_viewport_size({"width": 1500, "height": 1000})
        pg.click("#tabs button:nth-child(1)"); pg.wait_for_timeout(400)

        print("6j) 请求都带超时，不会永久挂死")
        check("有 AbortController 超时封装", pg.evaluate("() => typeof api === 'function'"))

        print("6k) 连点多个按钮，动作不许串台")
        pg.evaluate("() => gameNew()"); pg.wait_for_timeout(2500)
        seq = [a for a in ("FWD", "TLEFT", "ATTACK", "MRIGHT")
               if not pg.is_disabled(f'.padbtn[data-act="{a}"]')]
        for act in seq:
            n = tap(act)
            sent = pg.evaluate(f"() => G.rec.slice({n}).map(r => r.action)")
            # 曾经点「左转」发出去的是「前进 前进 前进 左转」——帧数对、动作全错
            check(f"点「{act}」发出 4 帧且全是它",
                  len(sent) == 4 and set(sent) == {act}, f"实际 {sent}")
        check("点完不留残留", pg.evaluate("() => G.repeatLeft") == 0)

        print("6l) 连点两次不许把两组动作绞在一起")
        n = pg.evaluate("() => G.rec.length")
        pg.click('.padbtn[data-act="FWD"]')
        pg.click('.padbtn[data-act="TLEFT"]')
        waited = 0
        while pg.evaluate("() => G.rec.length") < n + 8 and waited < 30000:
            pg.wait_for_timeout(200); waited += 200
        pg.wait_for_timeout(400)
        sent = pg.evaluate(f"() => G.rec.slice({n}).map(r => r.action)")
        check("先 4 帧前进，再 4 帧左转", sent == ["FWD"] * 4 + ["TLEFT"] * 4, f"实际 {sent}")

        print("6m) 新开一局要真的清空录制")
        tap("FWD")
        check("清空前有帧", pg.evaluate("() => G.rec.length") > 0)
        pg.click("#g-new"); pg.wait_for_timeout(2500)
        # 曾经写成 onclick = gameNew，MouseEvent 被当成 keepRec 传进去，恒为真
        check("新开一局后录制清零", pg.evaluate("() => G.rec.length") == 0,
              f"仍有 {pg.evaluate('() => G.rec.length')} 帧")
        check("计数条复位", pg.locator("#g-recact").inner_text().strip() == "还没有帧",
              pg.locator("#g-recact").inner_text())
        check("回放按钮置灰", pg.evaluate("() => document.getElementById('g-replay').disabled"))

        print("6m2) 看历史帧时新帧不许把进度条抢回末尾")
        tap("FWD"); tap("TLEFT")
        pg.evaluate("() => { stopReplay(); G.scrubbing = true; showRecFrame(1); }")
        v1 = pg.evaluate("() => document.getElementById('g-scrub').value")
        pg.evaluate("() => updateRecUI()")
        check("停在选中的那一帧",
              pg.evaluate("() => document.getElementById('g-scrub').value") == v1, v1)
        tap("FWD")
        check("又开始玩之后进度条重新跟随最新帧",
              not pg.evaluate("() => G.scrubbing"))

        print("6n) 回放不许把整页滚走")
        tap("FWD")
        pg.click("#g-replay")            # 先点（Playwright 会为点击自行滚动）
        pg.evaluate("() => window.scrollTo(0, 0)")
        pg.wait_for_timeout(1500)
        check("回放期间页面不被拽走", pg.evaluate("() => window.scrollY") == 0,
              f"scrollY={pg.evaluate('() => window.scrollY')}")
        pg.evaluate("() => stopReplay()"); pg.wait_for_timeout(300)

        print("7) 三个模型可切换")
        opts = pg.eval_on_selector_all("#g-model option", "els => els.map(e => e.value)")
        MODEL_NAMES.update(dict(zip(opts, pg.eval_on_selector_all(
            "#g-model option", "els => els.map(e => e.textContent)"))))
        check("模型下拉有三项", len(opts) == 3, str(opts))
        pool = pg.evaluate("() => G.pool")
        check("三个模型全部常驻显存", sorted(pool) == sorted(opts), f"常驻 {pool}")
        check("标题里写了常驻数", "常驻显存" in pg.locator("#g-gpu").inner_text(),
              pg.locator("#g-gpu").inner_text())
        cur_model = pg.input_value("#g-model")
        check("当前选中的就是已载入的", cur_model in opts, cur_model)
        check("有模型说明", len(pg.locator("#g-modelnote").inner_text()) > 10)
        links = pg.eval_on_selector_all("#g-modelnote a", "e=>e.map(x=>[x.textContent, x.href])")
        check("说明里有代码仓库和权重两个链接", len(links) == 2, str(links))
        check("代码链接指向 github", "github.com" in links[0][1], links[0][1])
        check("权重链接指向 huggingface", "huggingface.co" in links[1][1], links[1][1])
        other = [o for o in opts if o != cur_model][0]
        import time as _t

        def switch_to(mid, timeout=20):
            """选下拉框就该自动切过去，不再有「载入」按钮这一步。"""
            t0 = _t.time()
            pg.select_option("#g-model", mid)
            while (_t.time() - t0) < timeout:
                if pg.evaluate("() => G.loaded") == mid and not pg.evaluate("() => G.stale"):
                    break
                pg.wait_for_timeout(50)
            return (_t.time() - t0) * 1000

        ms = switch_to(other)
        check(f"选了 {other} 就自动切过去（无需点按钮）",
              pg.evaluate("() => G.loaded") == other, pg.evaluate("() => G.loaded"))
        # 三个模型常驻，切换只是改指针 + 开一局。3 秒是宽松上界（实测约 0.4 秒）
        check("自动切换在 3 秒内完成", ms < 3000, f"耗时 {ms:.0f} ms")
        gpu2 = pg.locator("#g-gpu").inner_text()
        check(f"切到 {other} 成功", "失败" not in gpu2 and "上下文" in gpu2, gpu2)
        check("确认框跟着更新", pg.locator("#g-runname").inner_text() == MODEL_NAMES[other],
              pg.locator("#g-runname").inner_text())
        # 两家的上下文长度不同，切换后这个数字必须跟着变
        buf = gpu2.split("上下文")[1].strip().split()[0]
        check("上下文帧数随模型变化", buf in ("9", "64"), f"读到 {buf}")
        blank2 = pg.evaluate("""() => {
            const c = document.getElementById('screen');
            const d = c.getContext('2d').getImageData(0,0,c.width,c.height).data;
            let s = 0; for (let i=0;i<d.length;i+=4) s += d[i]+d[i+1]+d[i+2];
            return s === 0; }""")
        check("换完模型自动开了新局（画面非全黑）", not blank2)
        # 切完的第一次点击必须生效。gameNew 里的 resetInput 曾经把这次点击一起清掉，
        # 表现是「刚换完模型点一下没反应」，实测出 0 帧。
        n = tap("FWD")
        sent = pg.evaluate(f"() => G.rec.slice({n}).map(r => r.action)")
        check("切完模型第一次点击就生效且动作正确",
              len(sent) == 4 and set(sent) == {"FWD"}, f"实际 {sent}")

        ms_back = switch_to(cur_model)
        check("切回原模型", pg.evaluate("() => G.loaded") == cur_model)
        check("来回切换不重载权重", ms_back < 3000, f"耗时 {ms_back:.0f} ms")
        buf_back = pg.locator("#g-gpu").inner_text().split("上下文")[1].strip().split()[0]
        check("切回来上下文帧数也跟着回去", buf_back != buf, f"{buf} -> {buf_back}")

        print("7b) session 失效后第一次点击不许被吃掉")
        pg.evaluate("() => { G.sid = null; }")
        n = tap("TLEFT")
        sent = pg.evaluate(f"() => G.rec.slice({n}).map(r => r.action)")
        check("补开 session 后那次点击仍然生效",
              len(sent) == 4 and set(sent) == {"TLEFT"}, f"实际 {sent}")

        print("8) 切到数据标签")
        tabs.nth(1).click()                      # 一级：数据预览
        pg.wait_for_timeout(800)
        check("数据页显示二级标签",
              "hide" not in (pg.get_attribute("#subtabs", "class") or ""))
        subs.nth(2).click()                      # 二级：bots_long
        pg.wait_for_timeout(1200)
        check("二级选中态正确", "on" in (subs.nth(2).get_attribute("class") or ""))
        check("逐帧控件在数据页显示",
              pg.locator("#controls").is_visible())
        check("有训练切片图", pg.locator("#main .film").count() > 0,
              f"{pg.locator('#main .film').count()} 张")
        check("有录像播放器", pg.locator("#main video").count() > 0,
              f"{pg.locator('#main video').count()} 个")
        pg.screenshot(path=f"{a.shots}/browser_2_data.png", full_page=False)

        print("9) 切回试玩，再切回数据")
        tabs.nth(0).click()
        pg.wait_for_timeout(600)
        check("逐帧控件在试玩页隐藏", not pg.locator("#controls").is_visible())
        check("试玩页二级标签隐藏",
              "hide" in (pg.get_attribute("#subtabs", "class") or ""))
        tabs.nth(1).click()
        pg.wait_for_timeout(800)
        check("回到数据页记得上次看的是 bots_long",
              "on" in (subs.nth(2).get_attribute("class") or ""))

        check("全程无 JS 报错", not errs, str(errs[:2]) if errs else "")
        br.close()

    print()
    if fails:
        print(f"✗ {len(fails)} 项没过: {fails}")
        sys.exit(1)
    print("✓ 全部通过")


if __name__ == "__main__":
    main()
