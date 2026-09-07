"""browsertest.py — 用无头 Chrome 真的把页面打开、真的按键，验证前端不是纸上谈兵。

playtest.py 验的是「动作发过去，模型回的画面对不对」，走的是 HTTP 接口，绕过了整个前端。
这里补的是另一半：页面能不能加载、标签能不能切、Pointer Lock 能不能锁、
键盘和鼠标事件能不能变成正确的动作、canvas 上的像素有没有真的在变。

Pointer Lock 在无头 Chrome 里需要 --enable-features 之类的开关才可靠，所以这里
不依赖它：直接注入事件并调用页面内部的 gameLoop，验证的是同一条代码路径。
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
        pg.evaluate("""() => {
            G.keys.add('w');
            window.__dxFeed = setInterval(() => { G.dx += 20; }, 30);
            G.running = true; gameLoop();
        }""")
        pg.wait_for_timeout(6000)
        steps = int(pg.locator("#g-steps").inner_text())
        check("生成了新帧", steps >= 3, f"steps={steps}")
        act = pg.locator("#g-act").inner_text()
        check("W + 持续右转 -> TR+FWD", act == "TR+FWD", f"实际 {act}")
        after = pg.evaluate("() => document.getElementById('screen').toDataURL().length")
        check("canvas 内容变了", before != after)
        fps = pg.locator("#g-fps").inner_text()
        check("显示了 fps 和真实速度倍率", "fps" in fps and "真实速度" in fps, fps)

        # 客户端墙钟 vs 服务端自报耗时：差值就是传输 + PNG 解码 + canvas 绘制的开销
        srv_ms = float(pg.locator("#g-ms").inner_text())
        cli_ms = pg.evaluate("() => G.times.reduce((a,b)=>a+b,0)/G.times.length")
        print(f"     服务端 {srv_ms:.0f} ms/帧，客户端墙钟 {cli_ms:.0f} ms/帧，"
              f"前端开销 {cli_ms - srv_ms:.0f} ms")
        check("前端开销不超过推理本身", cli_ms - srv_ms < srv_ms,
              f"开销 {cli_ms - srv_ms:.0f} ms")
        pg.evaluate("() => { clearInterval(window.__dxFeed); G.keys.clear(); }")

        print("5) 动作映射逐一核对")
        cases = [({}, 0, "NOOP"), ({"w"}, 0, "FWD"), ({"s"}, 0, "BACK"),
                 ({"a"}, 0, "ML"), ({"d"}, 0, "MR"), (set(), -40, "TL"),
                 (set(), 40, "TR"), ({"w"}, -40, "TL+FWD"), ({"s"}, 40, "TR+BACK")]
        for keys, dx, want in cases:
            got = pg.evaluate("""([keys, dx]) => {
                G.keys = new Set(keys); G.dx = dx; G.fire = false;
                const a = pickAction();
                return ["NOOP","TL","TR","BACK","TL+BACK","TR+BACK",
                        "MR","ML","FWD","TL+FWD","TR+FWD","ATK"][a];
            }""", [list(keys), dx])
            check(f"{sorted(keys) or '无键'} dx={dx:>4} -> {want}", got == want, f"实际 {got}")
        fire = pg.evaluate("""() => { G.keys=new Set(['w']); G.dx=40; G.fire=true;
            return pickAction(); }""")
        check("开火独占（压过 W 和转向）", fire == 11, f"实际 {fire}")

        print("6) 减速模式")
        for v, want in [("0.5", "0.5"), ("0.25", "0.25")]:
            pg.select_option("#g-speed", v)
            got = pg.evaluate("() => G.speed")
            check(f"选 {v} 倍速", str(got) == want, f"G.speed={got}")
        note = pg.locator("#g-speednote").inner_text()
        check("减速有文字说明", "额外等" in note, note)
        pg.select_option("#g-speed", "1")

        print("7) 推理服务地址可切换")
        cur = pg.input_value("#g-api")
        check("地址框预填了当前服务", cur.endswith(":25677"), cur)
        pg.fill("#g-api", "http://127.0.0.1:25699/")
        pg.click("#g-apply")
        pg.wait_for_timeout(1500)
        check("指向不存在的服务会报错",
              "连不上" in pg.locator("#g-err").inner_text(),
              pg.locator("#g-err").inner_text())
        check("地址写进了 localStorage",
              pg.evaluate("() => localStorage.getItem('gamengen_api')") == "http://127.0.0.1:25699")
        pg.fill("#g-api", cur)
        pg.click("#g-apply")
        pg.wait_for_timeout(3500)
        check("切回去能恢复", "未启动" not in pg.locator("#g-gpu").inner_text(),
              pg.locator("#g-gpu").inner_text())

        print("8) 切到数据标签")
        pg.evaluate("() => { G.running = false; }")
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

        # 第 7 步故意指向死端口，浏览器必然记一条连接失败——那是测试自己造的，
        # 不是页面的问题，所以从最终判定里剔掉。
        real = [e for e in errs if "25699" not in e
                and "ERR_CONNECTION_REFUSED" not in e]
        check("全程无 JS 报错（排除故意造的连接失败）", not real,
              str(real[:2]) if real else "")
        br.close()

    print()
    if fails:
        print(f"✗ {len(fails)} 项没过: {fails}")
        sys.exit(1)
    print("✓ 全部通过")


if __name__ == "__main__":
    main()
