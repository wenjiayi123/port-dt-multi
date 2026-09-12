from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser
import re
import unittest
from urllib.parse import unquote, urlsplit
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import server


# These fragments are public entry points used by the menu, standalone pages,
# and cross-module links. Renaming one must not silently strand an old link.
LEGACY_MODULE_IDS = {
    "exec-cockpit", "platform-map", "dev-ecosystem", "multiport-section",
    "esg-section", "cmp-section", "twin3d-section", "story-section",
    "kpi-section", "realtime-section", "twin-section", "strategy-exec-module",
    "rlops-section", "exec-wrap", "mas-section", "twinlab-section",
    "train-section", "deep-section", "yl-section", "hvac-section",
    "sbess-section", "be-section", "yc-section", "ai-trust-scenes",
    "monitoring-section", "opsx-section", "ext-section", "mlops-section",
    "g-section",
}
STANDALONE_ROUTES = {
    "/ops-copilot", "/v3?from=home", "/rl-panel", "/integration-hub",
    "/rl_future/rl_future_panel.html", "/docs",
}
VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
}


@dataclass
class Element:
    tag: str
    attrs: dict[str, str | None]
    ancestors: tuple[Element, ...]
    content: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return str(self.attrs.get("aria-label") or " ".join(self.content)).strip()

    def has_class(self, name: str) -> bool:
        return name in str(self.attrs.get("class") or "").split()


class Page(HTMLParser):
    def __init__(self, html: str):
        super().__init__(convert_charrefs=True)
        self.elements: list[Element] = []
        self.stack: list[Element] = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        element = Element(tag, dict(attrs), tuple(self.stack))
        self.elements.append(element)
        if tag not in VOID_TAGS:
            self.stack.append(element)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        for element in self.stack:
            element.content.append(data)

    @property
    def scripts(self) -> str:
        return "\n".join("".join(el.content) for el in self.elements if el.tag == "script")


def is_home_url(value: str) -> bool:
    url = urlsplit(value)
    return not url.scheme and not url.netloc and url.path == "/" and url.fragment in {"", "home-hero"}


def declared_button_destination(page: Page, element: Element) -> str | None:
    """Read the existing JS-only navigation controls' declared click target.

    This checks the HTTP-delivered navigation contract without running business
    scripts or training actions. Browser acceptance separately exercises clicks.
    """
    element_id = element.attrs.get("id")
    if not element_id:
        return None
    binding = re.search(
        rf"\$\(\s*(['\"])#{re.escape(element_id)}\1\s*\)\s*\??\.addEventListener"
        r"\(\s*(['\"])click\2\s*,",
        page.scripts,
    )
    if not binding:
        return None
    callback = re.split(r";\s*(?:\n|$)", page.scripts[binding.end():], maxsplit=1)[0]
    destination = re.search(
        r"(?:goBackTo|(?:window\.)?location\.(?:assign|replace))\s*\(\s*(['\"])(.*?)\1",
        callback,
    )
    return destination.group(2) if destination else None


class ModuleNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(server.app)
        cls.home_response = cls.client.get("/")
        cls.home = Page(cls.home_response.text)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def assert_accessible_control(self, element: Element):
        self.assertTrue(element.label, "Navigation control has no accessible label")
        self.assertNotIn("disabled", element.attrs)
        self.assertNotEqual(element.attrs.get("tabindex"), "-1")
        self.assertNotIn("hidden", element.attrs)
        self.assertNotEqual(element.attrs.get("aria-hidden"), "true")

    def test_main_page_and_navigation_assets_are_served_with_valid_types(self):
        self.assertEqual(self.home_response.status_code, 200)
        self.assertEqual(self.home_response.headers["content-type"].split(";")[0], "text/html")
        for filename, tag, attribute, media_types in (
            ("module_navigation.js", "script", "src", {"text/javascript", "application/javascript"}),
            ("module_navigation.css", "link", "href", {"text/css"}),
        ):
            with self.subTest(asset=filename):
                assets = [
                    el for el in self.home.elements
                    if el.tag == tag and urlsplit(el.attrs.get(attribute) or "").path.endswith("/" + filename)
                ]
                self.assertEqual(len(assets), 1, f"Expected one loaded {filename}")
                response = self.client.get(assets[0].attrs[attribute])
                self.assertEqual(response.status_code, 200)
                self.assertIn(response.headers["content-type"].split(";")[0], media_types)
                self.assertTrue(response.content.strip())

    def test_all_29_submenu_links_resolve_to_unique_enrolled_modules(self):
        links = [
            el for el in self.home.elements
            if el.tag == "a"
            and any(parent.attrs.get("id") == "panel-nav-primary" for parent in el.ancestors)
            and any(parent.has_class("nav-dropdown") for parent in el.ancestors)
        ]
        self.assertEqual(len(links), 29)
        ids = Counter(el.attrs.get("id") for el in self.home.elements if el.attrs.get("id"))
        targets = []
        for link in links:
            with self.subTest(label=link.label):
                self.assert_accessible_control(link)
                self.assertIn("data-module-page", link.attrs)
                href = urlsplit(link.attrs.get("href") or "")
                self.assertFalse(href.path)
                target = unquote(href.fragment)
                self.assertTrue(target)
                self.assertEqual(ids[target], 1, f"Missing or duplicated module #{target}")
                self.assertEqual(link.attrs.get("data-target-id") or target, target)
                targets.append(target)
        self.assertEqual(len(set(targets)), len(targets))
        self.assertEqual(set(targets), LEGACY_MODULE_IDS)

    def test_home_return_and_legacy_detail_anchors_remain_accessible(self):
        ids = Counter(el.attrs.get("id") for el in self.home.elements if el.attrs.get("id"))
        for anchor in ("home-hero", "exec-section", "exec-block"):
            self.assertEqual(ids[anchor], 1)
        home_links = [
            el for el in self.home.elements
            if el.tag == "a" and el.has_class("module-home-link")
        ]
        self.assertEqual(len(home_links), 1)
        self.assert_accessible_control(home_links[0])
        self.assertIn(home_links[0].attrs.get("href"), {"#home-hero", "/", "/#home-hero"})
        headings = [el for el in self.home.elements if el.attrs.get("id") == "module-page-title"]
        self.assertEqual(len(headings), 1)
        self.assertIn(headings[0].tag, {"h1", "h2"})
        self.assertTrue(headings[0].label)

    def test_every_standalone_menu_destination_exposes_a_home_control(self):
        routes = {
            el.attrs["data-route"] for el in self.home.elements
            if el.tag == "a" and el.has_class("nav-trigger") and el.attrs.get("data-route")
        }
        self.assertEqual(routes, STANDALONE_ROUTES)
        for route in sorted(routes):
            with self.subTest(route=route):
                response = self.client.get(route)
                if route == "/docs" and server.app.openapi_url is None:
                    self.assertEqual(response.status_code, 404)
                    continue
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["content-type"].split(";")[0], "text/html")
                page = Page(response.text)
                controls = []
                for el in page.elements:
                    if el.tag == "a" and is_home_url(el.attrs.get("href") or ""):
                        if el.attrs.get("target") in {None, "", "_self"}:
                            controls.append(el)
                    elif el.tag == "button" and re.search(r"(?:回|返回).*(?:首页|主界面)", el.label):
                        destination = declared_button_destination(page, el)
                        if destination and is_home_url(destination):
                            controls.append(el)
                self.assertTrue(controls, f"{route} has no same-tab return to the homepage")
                for control in controls:
                    self.assert_accessible_control(control)

    def test_future_deck_preserves_the_strategy_deep_link_beside_home(self):
        page = Page(self.client.get("/rl_future/rl_future_panel.html").text)
        targets = {el.attrs.get("href") for el in page.elements if el.tag == "a"}
        self.assertIn("/", targets)
        self.assertIn("/#strategy-exec-module", targets)

    def test_standalone_menu_matches_authoritative_home_navigation(self):
        response = self.client.get("/ui/module-menu")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"].split(";")[0], "text/html")
        self.assertIn("no-cache", response.headers["cache-control"])
        menu = Page(response.text)
        self.assertEqual(sum(el.attrs.get("id") == "panel-nav-primary" for el in menu.elements), 1)
        self.assertFalse(any(el.tag in {"script", "iframe", "main", "section"} for el in menu.elements))
        home_links = [
            el for el in self.home.elements if el.tag == "a"
            and any(parent.attrs.get("id") == "panel-nav-primary" for parent in el.ancestors)
        ]
        menu_links = [el for el in menu.elements if el.tag == "a"]
        expected = [("/" + el.attrs["href"] if el.attrs["href"].startswith("#") else el.attrs["href"], el.label) for el in home_links]
        self.assertEqual([(el.attrs["href"], el.label) for el in menu_links], expected)
        self.assertEqual(len(menu_links), len(LEGACY_MODULE_IDS) + len(STANDALONE_ROUTES))
        for link in menu_links:
            self.assertIn(link.attrs.get("target"), {None, "", "_self"})
            self.assertTrue(link.attrs["href"].startswith("/"))
            self.assertFalse(link.attrs["href"].startswith("//"))
        home_buttons = [el for el in menu.elements if el.attrs.get("data-direct-target")]
        self.assertEqual(len(home_buttons), 1)
        self.assertTrue(is_home_url(home_buttons[0].attrs["data-direct-target"]))

    def test_all_six_standalone_views_load_shared_navigation_resources(self):
        for route in sorted(STANDALONE_ROUTES):
            with self.subTest(route=route):
                response = self.client.get(route)
                if route == "/docs" and server.app.openapi_url is None:
                    self.assertEqual(response.status_code, 404)
                    continue
                self.assertEqual(response.status_code, 200)
                page = Page(response.text)
                for filename, tag, attr, types in (
                    ("standalone_navigation.js", "script", "src", {"text/javascript", "application/javascript"}),
                    ("standalone_navigation.css", "link", "href", {"text/css"}),
                ):
                    assets = [el for el in page.elements if el.tag == tag and urlsplit(el.attrs.get(attr) or "").path == "/static/" + filename]
                    self.assertEqual(len(assets), 1, f"{route}: expected one {filename}")
                    if tag == "script":
                        self.assertNotIn("defer", assets[0].attrs)
                        self.assertNotIn("async", assets[0].attrs)
                        self.assertTrue(any(parent.tag == "head" for parent in assets[0].ancestors))
                    asset = self.client.get(assets[0].attrs[attr])
                    self.assertEqual(asset.status_code, 200)
                    self.assertIn(asset.headers["content-type"].split(";")[0], types)
                    self.assertTrue(asset.content.strip())

    def test_missing_menu_source_is_reported_without_breaking_standalone_page(self):
        with patch.object(server, "_UI_INDEX") as source:
            source.exists.return_value = False
            self.assertEqual(self.client.get("/ui/module-menu").status_code, 503)
            page = Page(self.client.get("/ops-copilot").text)
            self.assertTrue(any(el.tag == "a" and is_home_url(el.attrs.get("href") or "") for el in page.elements))


if __name__ == "__main__":
    unittest.main()
