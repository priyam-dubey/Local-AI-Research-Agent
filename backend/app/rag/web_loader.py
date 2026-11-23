"""Fetch a public web page as plain text, with SSRF protection."""

from __future__ import annotations

import ipaddress
import socket
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx

from app.core.errors import BadRequestError, IngestionError

MAX_BYTES = 5 * 1024 * 1024
MAX_REDIRECTS = 3


def assert_public_url(url: str) -> None:
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise BadRequestError("Only http(s) URLs are allowed.")
    try:
        infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise BadRequestError("The URL's host could not be resolved.") from exc
    for info in infos:
        if not ipaddress.ip_address(info[4][0]).is_global:
            raise BadRequestError("URLs that resolve to private, loopback or link-local addresses are blocked.")


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.skip = 0
        self.parts: list[str] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "svg", "head"):
            self.skip += 1
        if tag == "title":
            self._in_title = True
        if tag in ("p", "br", "div", "li", "h1", "h2", "h3", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg", "head"):
            self.skip = max(0, self.skip - 1)
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skip:
            self.parts.append(data)


def fetch_url_text(url: str, transport: Optional[httpx.BaseTransport] = None) -> tuple[str, str]:
    """Returns (title, text). Re-validates every redirect hop; caps size; html/text only."""
    with httpx.Client(timeout=15, follow_redirects=False, transport=transport,
                      headers={"User-Agent": "LocalAIResearchAgent/2.0"}) as c:
        for _ in range(MAX_REDIRECTS + 1):
            assert_public_url(url)
            with c.stream("GET", url) as r:
                if r.is_redirect:
                    url = urljoin(url, r.headers.get("location", ""))
                    continue
                if r.status_code >= 400:
                    raise IngestionError(f"The page returned HTTP {r.status_code}.")
                ctype = r.headers.get("content-type", "")
                if not any(t in ctype for t in ("text/html", "text/plain", "application/xhtml")):
                    raise IngestionError("Only HTML or plain-text pages can be imported.")
                data = b""
                for chunk in r.iter_bytes():
                    data += chunk
                    if len(data) > MAX_BYTES:
                        raise IngestionError("The page is larger than 5 MB.")
                html = data.decode(r.encoding or "utf-8", errors="replace")
                if "text/plain" in ctype:
                    return urlparse(url).netloc, html
                parser = _Text()
                parser.feed(html)
                text = "\n".join(l.strip() for l in "".join(parser.parts).splitlines() if l.strip())
                return (parser.title.strip() or urlparse(url).netloc)[:120], text
    raise IngestionError("Too many redirects.")
