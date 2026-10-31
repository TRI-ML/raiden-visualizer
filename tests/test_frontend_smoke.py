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
           "datasets": [{"id": "demo", "label": "Demo webbed", "kind": "webbed", "num_episodes": 2, "num_tasks": 1, "total_hours": 0.1,
                         "cameras": ["cam_high"], "preview": None, "annotations": "n/a"}]}
TASK = {"task": "kitchen_put_spoon_seg48", "episodes": 2, "featured": 2, "prompt": "put the spoon in the drawer", "tags": ["bimanual", "yam"],
        "collected_start": None, "collected_end": None, "latest": "chunk_000__episode_000001"}
DETAIL = {"task": TASK["task"], "episode": "chunk_000__episode_000001", "instruction": "put the spoon in the drawer", "status": None,
          "cameras": [{"name": "cam_high", "has_video": True, "eyes": ["left"]}, {"name": "cam_left_wrist", "has_video": True, "eyes": ["left"]}],
          "robot": None, "metadata": {"duration_s": 16.1}, "shard_bounds_s": [1.5, 3.0], "calibration": None, "annotations": []}
# episode 3 carries subtask (Argus) labels: {t, text} transitions, as the yam / lerobot adapters emit them
ANNOTATED = {**DETAIL, "episode": "chunk_000__episode_000003",
             "annotations": [{"t": 0.5, "text": "reach for the spoon"}, {"t": 6.0, "text": "grasp the spoon"},
                             {"t": 11.0, "text": "place it in the drawer"}]}


def _route(route):
    url = route.request.url
    path = url.split("://", 1)[1].split("/", 1)[1].split("?")[0]
    body = None
    if path == "api/sources": body = {"sources": [{"id": "demo", "label": "Demo webbed", "kind": "webbed"}]}
    elif path == "api/catalog": body = CATALOG
    elif path in ("api/contrib", "api/raiden_teachers", "api/stats"): return route.fulfill(status=404, body="{}")
    elif path == "api/sources/demo/tasks": body = {"tasks": [TASK["task"]]}
    elif path == "api/sources/demo/overview": body = {"bucket": "b", "prefix": "p", "region": "us-west-2", "num_episodes": 2, "num_tasks": 1, "tasks": [TASK], "stations": []}
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
        except Exception as e:  # noqa: BLE001 - no Chromium bundle on this box
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
    assert page.locator("#brand-home").inner_text().split() == ["YAM", "Datasets"]
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
    page = _open(browser, site)
    page.wait_for_selector(".cat-card")
    assert page.evaluate("document.documentElement.dataset.env") == "local"
    assert page.locator('#env-seg [data-env="prod"]').is_disabled()
    assert page.locator(".teacher-card").is_visible()
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
    page.wait_for_selector("#ov-tag-filter .facet-chip")
    assert "bimanual" in page.locator("#ov-tag-filter").inner_text()
    assert "null" not in page.locator("#ov-task-list").inner_text()
    page.close()


def test_language_timeline_syncs_with_the_clock(browser, site):
    page = _open(browser, site, "#demo/kitchen_put_spoon_seg48/chunk_000__episode_000001")
    page.wait_for_selector("#camera-grid .cam-tile")
    assert not page.locator("#annotations-card").is_visible()       # no labels: the card stays hidden
    page.close()
    page = _open(browser, site, "#demo/kitchen_put_spoon_seg48/chunk_000__episode_000003")
    page.wait_for_selector("#lang-track .lang-seg")
    assert page.locator("#lang-track .lang-seg").count() == 3
    assert page.locator("#lang-now").inner_text() == ""              # t=0 is before the first label
    page.evaluate("drawAllCursors(1)")
    assert "reach" in page.locator("#lang-now").inner_text()
    page.evaluate("drawAllCursors(7)")                               # the transport tick drives the timeline
    assert "grasp" in page.locator("#lang-now").inner_text()
    assert page.locator("#lang-track .lang-seg.active").get_attribute("data-i") == "1"
    head_x = page.evaluate("parseFloat(document.querySelector('#lang-track .lang-head').style.left)")
    assert abs(head_x - 100 * 7 / 16.1) < 0.5
    page.locator("#lang-track .lang-seg").nth(2).click()             # clicking a segment seeks to its start
    assert "drawer" in page.locator("#lang-now").inner_text()
    page.screenshot(path=str(SHOTS / "argus_timeline.png"))


def test_task_dropdown_filters_and_selects(browser, site):
    """The task select is wrapped by dropdown.js: button face, filter box, Enter picks and fires change."""
    page = _open(browser, site, hash_="#demo")
    page.wait_for_selector("#task-select option", state="attached")
    btn = page.locator("#task-select + .dd-btn")
    assert btn.inner_text().strip() == TASK["task"]            # mirrors the selected option
    btn.click()
    page.wait_for_selector(".dd-pop:not([hidden]) .dd-opt")
    page.fill(".dd-search", "zzz")
    assert page.locator(".dd-empty").count() == 1
    page.fill(".dd-search", "spoon")
    assert page.locator(".dd-opt").count() == 1
    for theme in ("dark", "light"):
        page.evaluate(f"document.documentElement.setAttribute('data-theme','{theme}')")
        page.screenshot(path=str(SHOTS / f"dropdown_rv_{theme}.png"))
    page.evaluate("window._chg = 0; document.querySelector('#task-select').addEventListener('change', () => window._chg++)")
    page.evaluate("document.querySelector('#task-select').selectedIndex = -1")   # force a real change on Enter
    page.keyboard.press("Enter")
    page.wait_for_selector(".dd-pop[hidden]", state="attached")
    assert page.evaluate("window._chg") == 1
    assert page.evaluate("document.querySelector('#task-select').value") == TASK["task"]
    page.evaluate("document.querySelector('#task-select').value = ''")   # programmatic set mirrors onto the face
    assert btn.inner_text().strip() == ""
    page.close()
