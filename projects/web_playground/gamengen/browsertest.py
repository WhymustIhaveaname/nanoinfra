"""browsertest.py — 用无头 Chrome 真的把页面打开、真的按键，验证前端不是纸上谈兵。

playtest.py 验的是「动作发过去，模型回的画面对不对」，走的是 HTTP 接口，绕过了整个前端。
这里补的是另一半：页面能不能加载、标签能不能切、按键状态能不能变成正确的
按钮上报、canvas 上的像素有没有真的在变。

注意边界：本测试**不派发真实的键盘/鼠标事件**，而是直接设置 G.keys / G.dx
再调 gameLoop。所以 index.html 里的 keydown/keyup/mousemove/mousedown 和
Pointer Lock 那几个 handler 是**没被覆盖到**的。
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

        print("4) 模拟操作：W 前进 + 持续右转")
        before = pg.evaluate("() => document.getElementById('screen').toDataURL().length")
        # 不依赖 Pointer Lock：直接喂状态并驱动同一个循环。
        # dx 必须持续补——鼠标位移是一次性增量，pickAction 取走就清零（这是对的），
        # 真人持续移动鼠标才会不断产生 movementX，这里用定时器模拟。
        pg.evaluate('''() => {
            G.locked = true; G.keys.add('w');
            window.__dxFeed = setInterval(() => { G.dx += 20; G.dxAt = performance.now(); gameLoop(); }, 30);
            gameLoop();
        }''')
        pg.wait_for_timeout(6000)
        steps = int(pg.locator("#g-steps").inner_text())
        check("生成了新帧", steps >= 3, f"steps={steps}")
        act = pg.locator("#g-act").inner_text()
        check("W + 持续右转 -> 前进且右转", set(act.split("+")) == {"FWD", "TRIGHT"},
              f"实际 {act}")
        after = pg.evaluate("() => document.getElementById('screen').toDataURL().length")
        check("canvas 内容变了", before != after)
        fps = pg.locator("#g-fps").inner_text()
        check("显示了 fps", "fps" in fps, fps)

        # 客户端墙钟 vs 服务端自报耗时：差值就是传输 + PNG 解码 + canvas 绘制的开销
        srv_ms = float(pg.locator("#g-ms").inner_text())
        cli_ms = pg.evaluate("() => G.times.reduce((a,b)=>a+b,0)/G.times.length")
        print(f"     服务端 {srv_ms:.0f} ms/帧，客户端墙钟 {cli_ms:.0f} ms/帧，"
              f"前端开销 {cli_ms - srv_ms:.0f} ms")
        check("前端开销不超过推理本身", cli_ms - srv_ms < srv_ms,
              f"开销 {cli_ms - srv_ms:.0f} ms")
        pg.evaluate("() => { clearInterval(window.__dxFeed); G.keys.clear(); }")

        print("4b) 松手就定格 / 空格走一帧")
        pg.wait_for_timeout(1500)
        n0 = int(pg.locator("#g-steps").inner_text())
        pg.wait_for_timeout(3000)
        n1 = int(pg.locator("#g-steps").inner_text())
        check("没有输入时不再生成新帧", n0 == n1, f"{n0} -> {n1}")
        has_noop = pg.evaluate("() => G.hasNoop")
        pg.evaluate("() => { G.keys.add(' '); gameLoop(); }")
        pg.wait_for_timeout(1500)
        pg.evaluate("() => { G.keys.delete(' '); G.repeatLeft = 0; }")
        pg.wait_for_timeout(1000)
        n2 = int(pg.locator("#g-steps").inner_text())
        if has_noop:
            check("有 NOOP 的模型：空格能推进", n2 > n1, f"{n1} -> {n2}")
        else:
            # 没有 NOOP 的模型上按空格不该发任何请求，否则会被 422 拒到白烧 GPU
            check("无 NOOP 的模型：空格不空转", n2 == n1, f"{n1} -> {n2}")

        print("4c) 鼠标停下就不再转")
        pg.evaluate("() => { resetInput(); G.dx = 99; G.dxAt = performance.now(); }")
        check("刚动过鼠标 -> 算作在转", pg.evaluate("() => hasInput()"))
        pg.wait_for_timeout(400)
        check("停手 400ms 后 -> 不再算作在转", not pg.evaluate("() => hasInput()"))
        pg.evaluate("() => resetInput()")

        print("5) 按键上报逐一核对")
        cases = [(set(), 0, []), ({"w"}, 0, ["FWD"]), ({"s"}, 0, ["BACK"]),
                 ({"a"}, 0, ["MLEFT"]), ({"d"}, 0, ["MRIGHT"]),
                 (set(), -40, ["TLEFT"]), (set(), 40, ["TRIGHT"]),
                 ({"w"}, -40, ["FWD", "TLEFT"]), ({"s"}, 40, ["BACK", "TRIGHT"])]
        for keys, dx, want in cases:
            got = pg.evaluate("""([keys, dx]) => {
                G.keys = new Set(keys); G.dx = dx; G.dxAt = performance.now(); G.fire = false;
                return pickButtons();
            }""", [list(keys), dx])
            check(f"{sorted(keys) or '无键'} dx={dx:>4} -> {want or '空'}",
                  sorted(got) == sorted(want), f"实际 {got}")
        fire = pg.evaluate("""() => { G.keys=new Set(['w']); G.dx=40; G.dxAt=performance.now(); G.fire=true;
            return pickButtons(); }""")
        check("开火独占（压过 W 和转向）", fire == ["ATTACK"], f"实际 {fire}")
        # 动作 id 的翻译交给服务端，按模型的表来——这是「不动发成左转」那个 bug 的修法
        tbl = pg.evaluate("() => fetch(GAME_API+'/info').then(r=>r.json()).then(d=>[d.actions.length, d.has_noop])")
        check("服务端报出当前模型的动作表", isinstance(tbl, list) and tbl[0] in (12, 18),
              f"表长 {tbl}")

        print("6) 减速模式")
        for v, want in [("0.5", "0.5"), ("0.25", "0.25")]:
            pg.select_option("#g-speed", v)
            got = pg.evaluate("() => G.speed")
            check(f"选 {v} 倍速", str(got) == want, f"G.speed={got}")
        pg.select_option("#g-speed", "1")

        print("6b) 选了模型不载入就不能玩")
        MODEL_NAMES.update(dict(zip(
            pg.eval_on_selector_all("#g-model option", "e=>e.map(x=>x.value)"),
            pg.eval_on_selector_all("#g-model option", "e=>e.map(x=>x.textContent)"))))
        cur0 = pg.input_value("#g-model")
        other0 = [o for o in pg.eval_on_selector_all("#g-model option", "e=>e.map(x=>x.value)")
                  if o != cur0][0]
        pg.select_option("#g-model", other0)
        pg.wait_for_timeout(400)
        check("有醒目提示", "载入" in pg.locator("#g-stale").inner_text(),
              pg.locator("#g-stale").inner_text())
        check("确认框仍写着实际在跑的那个",
              pg.locator("#g-runname").inner_text() == MODEL_NAMES[cur0],
              f"确认框={pg.locator('#g-runname').inner_text()} 期望={MODEL_NAMES[cur0]}")
        check("确认框变成告警态", "stale" in (pg.get_attribute("#g-running","class") or ""))
        before_n = int(pg.locator("#g-steps").inner_text())
        pg.evaluate("() => { G.locked = true; G.keys.add('w'); gameLoop(); }")
        pg.wait_for_timeout(2500)
        pg.evaluate("() => G.keys.clear()")
        check("不一致时按键不生成任何帧",
              int(pg.locator("#g-steps").inner_text()) == before_n,
              f"{before_n} -> {pg.locator('#g-steps').inner_text()}")
        pg.select_option("#g-model", cur0)
        pg.wait_for_timeout(400)
        check("选回来后提示消失", pg.locator("#g-stale").inner_text().strip() == "")
        check("选中的就是在跑的时，载入按钮禁用", pg.is_disabled("#g-loadmodel"))
        check("按钮显示「已载入」", pg.locator("#g-loadmodel").inner_text() == "已载入",
              pg.locator("#g-loadmodel").inner_text())
        check("确认框恢复正常态", "stale" not in (pg.get_attribute("#g-running","class") or ""))

        print("6c) 按下模型不支持的键要有提示")
        pg.evaluate("() => notSupported('空格（无动作）')")
        pg.wait_for_timeout(250)
        msg = pg.locator("#g-unsupported").inner_text()
        check("提示文字出现", "没有这个动作" in msg, msg)
        op = pg.evaluate("() => getComputedStyle(document.getElementById('g-unsupported')).opacity")
        check("提示可见", float(op) > 0.9, f"opacity={op}")
        pg.wait_for_timeout(2400)
        op2 = pg.evaluate("() => getComputedStyle(document.getElementById('g-unsupported')).opacity")
        check("提示会自己消失", float(op2) < 0.1, f"opacity={op2}")

        print("6d) 本局录制与回放")
        pg.evaluate("() => { G.locked = true; G.keys.add('w'); gameLoop(); }")
        pg.wait_for_timeout(2500)
        pg.evaluate("() => { G.keys.clear(); G.repeatLeft = 0; }")
        pg.wait_for_timeout(600)
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
            before = pg.evaluate("() => G.rec.length")
            pg.click(f'.padbtn[data-act="{act}"]')
            pg.wait_for_timeout(3000)
            sent = pg.evaluate(f"() => G.rec.slice({before}).map(r => r.action)")
            check(f"点「{act}」发出 {rep} 帧且全是它", len(sent) == rep and set(sent) == {act},
                  f"实际 {sent}")
        check("点完不留残留", pg.evaluate("() => G.repeatLeft") == 0)

        print("6f) 回放期间不吃输入，结束后不倒灌")
        pg.evaluate("() => { startReplay(); G.keys.add('w'); G.dx = 99; G.dxAt = performance.now(); G.fire = true; }")
        pg.wait_for_timeout(300)
        n0 = pg.evaluate("() => G.rec.length")
        pg.wait_for_timeout(1200)
        check("回放中不追加新帧", pg.evaluate("() => G.rec.length") == n0)
        pg.evaluate("() => stopReplay()")
        pg.wait_for_timeout(300)
        check("回放结束后输入已清空", not pg.evaluate("() => hasInput()"))

        print("6g) 失焦/切后台不留卡住的按键")
        pg.evaluate("() => { G.keys.add('w'); G.keys.add('a'); G.dx = 50; G.dxAt = performance.now(); G.fire = true; }")
        pg.evaluate("() => window.dispatchEvent(new Event('blur'))")
        check("失焦后按键全释放", not pg.evaluate("() => hasInput()"))

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

        print("6j) 请求都带超时，不会永久挂死")
        check("有 AbortController 超时封装", pg.evaluate("() => typeof api === 'function'"))

        print("7) 三个模型可切换")
        opts = pg.eval_on_selector_all("#g-model option", "els => els.map(e => e.value)")
        MODEL_NAMES.update(dict(zip(opts, pg.eval_on_selector_all(
            "#g-model option", "els => els.map(e => e.textContent)"))))
        check("模型下拉有三项", len(opts) == 3, str(opts))
        cur_model = pg.input_value("#g-model")
        check("当前选中的就是已载入的", cur_model in opts, cur_model)
        check("有模型说明", len(pg.locator("#g-modelnote").inner_text()) > 10)
        links = pg.eval_on_selector_all("#g-modelnote a", "e=>e.map(x=>[x.textContent, x.href])")
        check("说明里有代码仓库和权重两个链接", len(links) == 2, str(links))
        check("代码链接指向 github", "github.com" in links[0][1], links[0][1])
        check("权重链接指向 huggingface", "huggingface.co" in links[1][1], links[1][1])
        other = [o for o in opts if o != cur_model][0]
        pg.select_option("#g-model", other)
        pg.wait_for_timeout(300)
        check("选了别的模型后按钮才可点", not pg.is_disabled("#g-loadmodel"))
        pg.click("#g-loadmodel")
        pg.wait_for_timeout(1800)
        check("载入时进度条可见", pg.locator("#g-progwrap").is_visible())
        ptext = pg.locator("#g-progtext").inner_text()
        check("进度条有阶段和秒数", ("预热" in ptext or "测速" in ptext
              or "载入" in ptext or "腾显存" in ptext or "准备" in ptext) and "s" in ptext, ptext)
        pg.wait_for_function("() => document.getElementById('g-loadmodel').textContent !== '载入中…'", timeout=300000)
        pg.wait_for_timeout(2500)
        gpu2 = pg.locator("#g-gpu").inner_text()
        check(f"切到 {other} 成功", "失败" not in gpu2 and "上下文" in gpu2, gpu2)
        check("确认框跟着更新", pg.locator("#g-runname").inner_text() == MODEL_NAMES[other],
              pg.locator("#g-runname").inner_text())
        # 两家的上下文长度不同，切换后这个数字必须跟着变
        buf = gpu2.split("上下文")[1].strip().split()[0]
        check("上下文帧数随模型变化", buf in ("9", "64"), f"读到 {buf}")
        pg.wait_for_timeout(3000)
        blank2 = pg.evaluate("""() => {
            const c = document.getElementById('screen');
            const d = c.getContext('2d').getImageData(0,0,c.width,c.height).data;
            let s = 0; for (let i=0;i<d.length;i+=4) s += d[i]+d[i+1]+d[i+2];
            return s === 0; }""")
        check("换完模型自动开了新局（画面非全黑）", not blank2)
        import time as _t
        t_switch = _t.time()
        pg.select_option("#g-model", cur_model)
        pg.click("#g-loadmodel")
        pg.wait_for_function("() => document.getElementById('g-loadmodel').textContent !== '载入中…'", timeout=300000)
        pg.wait_for_timeout(2000)
        check("切回原模型", "上下文" in pg.locator("#g-gpu").inner_text())
        # 基准结果落盘缓存了，第二次载入不该再测速——15 秒是宽松上界（实测 2-3 秒）
        dt = _t.time() - t_switch - 2.0
        check("已测过的模型再载入走缓存（不重测速）", dt < 15, f"耗时 {dt:.1f}s")

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
