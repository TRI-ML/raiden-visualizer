"""Frontend smoke test: the static app against a mocked /api (no S3, no backend).

Serves static/ from a plain http.server thread and answers every /api/* request from the
browser with fixture JSON via Playwright's request interception, so it runs anywhere the
Chromium bundle is installed (skips otherwise).
"""
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pw = pytest.importorskip("playwright.sync_api")

STATIC = Path(__file__).resolve().parents[1] / "static"
SHOTS = Path("/tmp")

CATALOG = {"aggregate": {"num_datasets": 1, "total_episodes": 2, "total_tasks": 1, "total_hours": 0.1, "with_annotations": 0, "building": 0},
           "datasets": [{"id": "demo", "label": "Demo webbed", "kind": "webbed", "origin": "open", "embodiments": ["YAM", "Hand"], "num_episodes": 2, "num_tasks": 1, "total_hours": 0.1,
                         "cameras": ["cam_high"], "preview": None, "annotations": "n/a"}]}
# the persisted stats pass (stats_v2): per-episode records + the summary the tiles read; nothing is computed on request
STATS = {"num_episodes": 2, "total_episodes": 2, "scanned": 2, "sampled": False, "built_at": 1.0,
         "episodes": [{"task": "kitchen_put_spoon_seg48", "episode": "chunk_000__episode_000001", "duration_s": 16.1, "has_annotations": False},
                      {"task": "kitchen_put_spoon_seg48", "episode": "chunk_000__episode_000003", "duration_s": 20.0, "has_annotations": True}],
         "summary": {"hours": 0.0, "estimated": False, "median_duration_s": 18.1, "episodes_per_task_median": 2, "unique_tasks": 1,
                     "annotated_episodes": 1}}
TASK = {"task": "kitchen_put_spoon_seg48", "episodes": 2, "featured": 2, "prompt": "put the spoon in the drawer", "tags": ["bimanual", "yam"],
        "collected_start": None, "collected_end": None, "latest": "chunk_000__episode_000001"}
DETAIL = {"task": TASK["task"], "episode": "chunk_000__episode_000001", "instruction": "put the spoon in the drawer", "status": None,
          "cameras": [{"name": "cam_high", "has_video": True, "eyes": ["left"]}, {"name": "cam_left_wrist", "has_video": True, "eyes": ["left"]}],
          "robot": None, "metadata": {"duration_s": 16.1}, "shard_bounds_s": [1.5, 3.0], "calibration": None, "annotations": []}
# a YAM-style trajectory block with its server-built legend (raiden_viz.embodiments.joint_legend)
from raiden_viz import embodiments as _emb

_ROBOT = {"keys": ["follower_l_joint_pos", "follower_l_gripper_pos"], "time": [0.0, 1.0, 2.0],
          "signals": {"follower_l_joint_pos": {"dims": 6, "series": [[0.1 * i] * 6 for i in range(3)], "min": 0.0, "max": 0.2},
                      "follower_l_gripper_pos": {"dims": 1, "series": [[0.0], [0.5], [1.0]], "min": 0.0, "max": 1.0}},
          "summary": {"num_steps": 3, "duration_s": 2.0, "hz": 1.5}}
_ROBOT["joint_legend"] = _emb.joint_legend("raiden", _ROBOT)
_ROBOT["joint_legend"]["layout"] = _emb.layout("raiden", _ROBOT)
DETAIL["robot"] = _ROBOT
CONVENTION = _emb.convention("raiden")
# episode 3: a bimanual eef32 (UW YAM) trajectory → the two-column arm grid
_EEF = {"keys": ["state"], "time": [0.0, 1.0, 2.0],
        "signals": {"state": {"dims": 32, "series": [[0.1 * i] * 3 + [1, 0, 0, 0, 1, 0] + [0.2 * i] * 3 + [1, 0, 0, 0, 1, 0] + [0.5, 0.5] + [0.01 * i] * 12 for i in range(3)],
                              "min": 0.0, "max": 1.0}},
        "summary": {"num_steps": 3, "duration_s": 2.0, "hz": 1.5}}
_EEF["signals"]["action"] = dict(_EEF["signals"]["state"])
_EEF["keys"] = ["state", "action"]
_EEF["joint_legend"] = _emb.joint_legend("webbed", _EEF)
_EEF["joint_legend"]["layout"] = _emb.layout("webbed", _EEF)
# episode 3 carries subtask (Argus) labels: {t, text} transitions, as the yam / lerobot adapters emit them
ANNOTATED = {**DETAIL, "episode": "chunk_000__episode_000003", "robot": _EEF,
             "annotations": [{"t": 0.5, "text": "reach for the spoon"}, {"t": 6.0, "text": "grasp the spoon"},
                             {"t": 11.0, "text": "place it in the drawer"}]}


def _route(route):
    url = route.request.url
    path = url.split("://", 1)[1].split("/", 1)[1].split("?")[0]
    body = None
    if path == "api/sources": body = {"sources": [{"id": "demo", "label": "Demo webbed", "kind": "webbed", "origin": "open", "embodiments": ["YAM", "Hand"]}]}
    elif path == "api/catalog": body = CATALOG
    elif path in ("api/contrib", "api/raiden_teachers"): return route.fulfill(status=404, body="{}")
    elif path == "api/sources/demo/stats": body = STATS
    elif path == "api/sources/demo/tasks": body = {"tasks": [TASK["task"]]}
    elif path == "api/sources/demo/overview": body = {"bucket": "b", "prefix": "p", "region": "us-west-2", "num_episodes": 2, "num_tasks": 1, "tasks": [TASK], "stations": [],
                                                       "state_convention": CONVENTION}
    elif path.endswith("/episodes"): body = {"episodes": ["chunk_000__episode_000001", "chunk_000__episode_000003"], "total": 2, "featured": False}
    elif path.endswith("/episode-facts"): body = {"facts": {}}
    elif path.endswith("/episodes/chunk_000__episode_000001"): body = DETAIL
    elif path.endswith("/episodes/chunk_000__episode_000003"): body = ANNOTATED
    elif path.endswith("/video") or "/video?" in url: return route.fulfill(status=404, body="")
    elif path.startswith("api/"): body = {}
    else: return route.continue_()
    route.fulfill(status=200, content_type="application/json", body=json.dumps(body))


@pytest.fixture(scope="module")
def site():
    handler = partial(SimpleHTTPRequestHandler, directory=str(STATIC))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture(scope="module")
def browser():
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:
            pytest.skip(f"chromium not available: {e}")
        yield b
        b.close()


def _open(browser, site, hash_="", query=""):
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.route("**/api/**", _route)
    page.goto(f"{site}/index.html{query}{hash_}", wait_until="domcontentloaded")
    return page


def test_catalog_renders_and_theme_toggle_persists(browser, site):
    page = _open(browser, site)
    page.wait_for_selector(".cat-card")
    assert "Demo webbed" in page.locator(".cat-card-title").first.inner_text()
    assert page.evaluate("document.documentElement.dataset.theme") in ("dark", "light")
    page.evaluate("applyTheme('dark')")
    page.screenshot(path=str(SHOTS / "rv_dark_catalog.png"))
    page.click('#theme-seg [data-theme="light"]')
    assert page.evaluate("document.documentElement.dataset.theme") == "light"
    assert "active" in page.get_attribute('#theme-seg [data-theme="light"]', "class")
    assert page.evaluate("localStorage.getItem('rv_theme')") == "light"
    page.screenshot(path=str(SHOTS / "rv_light_catalog.png"))
    page.reload(wait_until="domcontentloaded")
    assert page.evaluate("document.documentElement.dataset.theme") == "light"   # persisted
    # every colour goes through a token: the light sheet changes the page background
    bg = page.evaluate("getComputedStyle(document.body).backgroundColor")
    assert bg == "rgb(245, 246, 249)", bg
    assert page.locator("#brand-home").inner_text().split() == ["Robot", "Datasets"]
    page.close()


def test_env_segment_and_activity_panels(browser, site):
    # prod is the lean site: Prod lit, fleet-wide activity panels hidden; dev: Dev lit, panels shown; local (no hostname):
    # segment disabled, full feature set like dev
    page = _open(browser, site, query="?env=prod")
    page.wait_for_selector(".cat-card")
    assert "active" in page.get_attribute('#env-seg [data-env="prod"]', "class")
    assert page.locator(".contrib-card").first.is_hidden()
    assert page.locator(".teacher-card").is_hidden()
    page.screenshot(path=str(SHOTS / "rv_header_prod.png"))
    page.close()
    page = _open(browser, site, query="?env=dev")
    page.wait_for_selector(".cat-card")
    assert "active" in page.get_attribute('#env-seg [data-env="dev"]', "class")
    assert page.locator(".contrib-card").first.is_visible()
    assert page.locator(".teacher-card").is_visible()
    page.screenshot(path=str(SHOTS / "rv_header_dev.png"))
    page.close()
    # no query, no stored choice: dev is the default off the hosted prod host; clicking Prod persists and reloads lean
    page = _open(browser, site)
    page.wait_for_selector(".cat-card")
    assert page.evaluate("document.documentElement.dataset.env") == "dev"
    assert page.locator(".teacher-card").is_visible()
    page.click('#env-seg [data-env="prod"]')
    page.wait_for_selector(".cat-card")
    assert page.evaluate("document.documentElement.dataset.env") == "prod"
    assert page.locator(".teacher-card").is_hidden()
    page.evaluate("localStorage.removeItem('rv_env')")
    page.close()


def test_episode_view_renders_camera_grid_and_generic_names(browser, site):
    page = _open(browser, site, "#demo/kitchen_put_spoon_seg48/chunk_000__episode_000001")
    page.wait_for_selector("#camera-grid .cam-tile")
    assert page.locator("#camera-grid .cam-tile").count() == 2
    assert "put the spoon" in page.locator("#ep-instruction").inner_text()
    # the sidebar episode rows read "episode 1", not the raw chunk path; the tag filter shows the task's tags
    assert "episode 1" in page.locator("#episode-list li").first.inner_text()
    page.evaluate("applyTheme('dark')")
    page.screenshot(path=str(SHOTS / "rv_dark_episode.png"))
    page.evaluate("applyTheme('light')")
    page.screenshot(path=str(SHOTS / "rv_light_episode.png"))
    page.evaluate("showOverview()")
    page.wait_for_selector(".ov-task-row")
    assert page.locator("#ov-tag-filter").count() == 0   # the tag chip row is gone (PR #108)
    assert "null" not in page.locator("#ov-task-list").inner_text()
    page.close()


def test_episode_switch_keeps_the_old_content_until_the_new_one_lands(browser, site):
    # no loading screen: the header takes the new episode's name at once and the previous cameras stay in the DOM
    # while the (slow) detail request is in flight; the slim busy bar is the only indicator
    page = _open(browser, site, "#demo/kitchen_put_spoon_seg48/chunk_000__episode_000001")
    page.wait_for_selector("#camera-grid .cam-tile")
    old_tiles = page.evaluate("[...document.querySelectorAll('#camera-grid .cam-tile')].map(t => t.outerHTML.length)")

    held = []   # the detail request is held open until we have looked at the screen (a sleep would block the dispatcher)
    page.route("**/episodes/chunk_000__episode_000003", lambda route: held.append(route))
    page.evaluate("() => { selectEpisode('chunk_000__episode_000003'); }")   # not awaited: the request is held
    page.wait_for_timeout(300)
    assert page.locator("#ep-instruction").inner_text() == "episode 3"          # never "Loading…"
    assert page.evaluate("document.documentElement.classList.contains('busy')")   # the bar is lit while in flight
    assert page.locator("#camera-grid .cam-tile").count() == len(old_tiles)       # previous episode still on screen
    assert page.locator("#lang-card .lang-row, #lang-list > *").count() == 0
    page.screenshot(path=str(SHOTS / "rv_switch_t0.png"))
    held[0].fulfill(status=200, content_type="application/json", body=json.dumps(ANNOTATED))
    page.wait_for_selector("#lang-list > *")
    assert "put the spoon" in page.locator("#ep-instruction").inner_text()
    assert not page.evaluate("document.documentElement.classList.contains('busy')")
    page.screenshot(path=str(SHOTS / "rv_switch_t1.png"))
    page.close()


def test_language_timeline_syncs_with_the_clock(browser, site):
    page = _open(browser, site, "#demo/kitchen_put_spoon_seg48/chunk_000__episode_000001")
    page.wait_for_selector("#camera-grid .cam-tile")
    assert not page.locator("#lang-track").is_visible()                # no labels: no track, no Language card
    assert not page.locator("#lang-card").is_visible()
    assert "repeat(2" in page.evaluate("document.querySelector('#camera-grid').style.gridTemplateColumns")   # 1×N row
    page.close()
    page = _open(browser, site, "#demo/kitchen_put_spoon_seg48/chunk_000__episode_000003")
    page.wait_for_selector("#lang-track .lang-seg")
    segs, rows = page.locator("#lang-track .lang-seg"), page.locator("#lang-list .lang-row")
    assert segs.count() == 3 and rows.count() == 3
    assert all(segs.nth(i).inner_text() == "" for i in range(3))      # colored blocks only on the track
    colors = [segs.nth(i).evaluate("n => n.style.background") for i in range(3)]
    assert all(colors) and len(set(colors)) == 3                       # one color per distinct label
    assert [rows.nth(i).locator(".lang-swatch").evaluate("n => n.style.background") for i in range(3)] == colors
    assert "grasp" in rows.nth(1).inner_text() and "6.0s" in rows.nth(1).inner_text()
    # the track sits in the scrub-wrap, so it spans exactly the scrubber's width
    assert page.evaluate("document.querySelector('#lang-track').getBoundingClientRect().width") == \
        page.evaluate("document.querySelector('#scrubber').getBoundingClientRect().width")
    assert rows.locator(".active").count() == 0                        # t=0 is before the first label
    page.evaluate("drawAllCursors(7)")                                 # the transport tick drives both views
    assert page.locator("#lang-track .lang-seg.active").get_attribute("data-i") == "1"
    assert page.locator("#lang-list .lang-row.active").get_attribute("data-i") == "1"
    head_x = page.evaluate("parseFloat(document.querySelector('#lang-track .lang-head').style.left)")
    assert abs(head_x - 100 * 7 / 16.1) < 0.5
    rows.nth(2).click()                                                # a row click seeks to the segment START
    assert page.locator("#lang-track .lang-seg.active").get_attribute("data-i") == "2"
    assert page.locator("#time-label").inner_text().startswith("11.0s")
    segs.nth(0).click()                                                # so does a track block
    assert page.locator("#time-label").inner_text().startswith("0.5s")
    # dragging the scrubber fires `input` events; each one moves the clock and the active segment
    page.evaluate("state.duration = 16.1; const s = document.querySelector('#scrubber'); s.value = 450; s.dispatchEvent(new Event('input'))")
    assert page.locator("#time-label").inner_text().startswith("7.2s")
    assert page.locator("#lang-list .lang-row.active").get_attribute("data-i") == "1"
    page.screenshot(path=str(SHOTS / "argus_timeline.png"))
    page.close()


def test_tab_icon_is_wired(browser, site):
    # the static bimanual-arm favicon: icon link present and the SVG served
    page = _open(browser, site)
    page.wait_for_selector(".cat-card")
    assert page.locator('link[rel="icon"][type="image/svg+xml"]').count() == 1
    assert page.request.get(f"{site}/favicon.svg").status == 200
    page.close()


def test_axis_legend_popover_and_convention_card(browser, site):
    # overview: the one-line state convention, expandable
    page = _open(browser, site, hash_="#demo/kitchen_put_spoon_seg48")
    page.wait_for_selector("#ov-convention:not([hidden])")
    assert page.locator("#conv-summary").inner_text().startswith("Joint space")
    page.click("#ov-convention summary")
    assert "FRAME" in page.locator("#conv-body").inner_text().upper()
    # episode: no "Robot trajectories" title; the ? lists every axis of the plotted signals
    page.close()
    page = _open(browser, site, hash_="#demo/kitchen_put_spoon_seg48/chunk_000__episode_000001")
    page.wait_for_selector("#plots .plot-block")
    assert page.locator(".plot-card h3").count() == 0
    assert "3 steps" in page.locator("#plot-summary").inner_text()
    assert page.locator("#plot-help-pop").is_hidden()
    page.click("#plot-help")
    page.wait_for_selector("#plot-help-pop:not([hidden])")
    assert page.locator("#plot-help-pop table").count() == 2          # gripper (1 row) + joints (6 rows)
    assert page.locator("#plot-help-pop table tr").count() == 7
    joints = page.locator("#plot-help-pop table").nth(1).locator("tr")
    assert joints.count() == 6 and "joint1" in joints.nth(0).inner_text() and "rad" in joints.nth(0).inner_text()
    assert "0 = closed" in page.locator("#plot-help-pop").inner_text()
    # series legend shows the short joint names, the title the unit
    assert "joint1" in page.locator("#plots .legend").first.inner_text()
    titles = page.locator("#plots .plot-title")
    assert "0–1" in titles.nth(0).inner_text() and "rad" in titles.nth(1).inner_text()   # gripper first, then joints
    page.locator(".plot-card").scroll_into_view_if_needed()
    page.screenshot(path=str(SHOTS / "rv_joint_help.png"))
    page.close()


def test_bimanual_ee_episode_renders_the_arm_grid(browser, site):
    page = _open(browser, site, hash_="#demo/kitchen_put_spoon_seg48/chunk_000__episode_000003")
    page.wait_for_selector("#plots .plot-grid")
    cols = page.evaluate("getComputedStyle(document.querySelector('#plots .plot-grid')).gridTemplateColumns.split(' ').length")
    assert cols == 2
    heads = page.locator("#plots .plot-arm-head").all_inner_texts()
    assert [h.upper() for h in heads] == ["LEFT ARM", "RIGHT ARM", "LEFT ARM", "RIGHT ARM"]
    titles = page.locator("#plots .plot-title").all_inner_texts()
    assert any(x.startswith("position") for x in titles) and any(x.startswith("rotation") and "deg" in x for x in titles)
    assert any(x.startswith("gripper") for x in titles) and any(x.startswith("joints") for x in titles)
    assert not any("state ·" in x or "action ·" in x for x in titles)
    # state and action are stacked subsections, each with its own arm grid
    assert [h.upper() for h in page.locator("#plots .plot-section-head").all_inner_texts()] == ["STATE", "ACTION"]
    assert page.locator("#plots .plot-grid").count() == 2
    # rotation panel: roll/pitch/yaw legend, raw toggle swaps to the 6 rot6d components
    rot = page.locator("#plots .plot-block", has_text="rotation").first
    assert "roll" in rot.locator(".legend").inner_text()
    rot.locator(".raw-toggle").click()
    assert rot.locator(".legend > span").count() == 6
    page.locator(".plot-card").scroll_into_view_if_needed()
    page.screenshot(path=str(SHOTS / "rv_traj_sections.png"))
    page.close()
