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


def _open(browser, site, hash_=""):
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.route("**/*", _route) if False else page.route("**/api/**", _route)
    page.goto(f"{site}/index.html{hash_}", wait_until="domcontentloaded")
    return page


def test_catalog_renders_and_theme_toggle_persists(browser, site):
    page = _open(browser, site)
    page.wait_for_selector(".cat-card")
    assert "Demo webbed" in page.locator(".cat-card-title").first.inner_text()
    assert page.evaluate("document.documentElement.dataset.theme") in ("dark", "light")
    page.evaluate("applyTheme('dark')")
    page.screenshot(path=str(SHOTS / "rv_dark_catalog.png"))
    page.click("#theme-toggle")
    assert page.evaluate("document.documentElement.dataset.theme") == "light"
    assert page.evaluate("localStorage.getItem('rv_theme')") == "light"
    page.screenshot(path=str(SHOTS / "rv_light_catalog.png"))
    page.reload(wait_until="domcontentloaded")
    assert page.evaluate("document.documentElement.dataset.theme") == "light"   # persisted
    # every colour goes through a token: the light sheet changes the page background
    bg = page.evaluate("getComputedStyle(document.body).backgroundColor")
    assert bg == "rgb(245, 246, 249)", bg
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
