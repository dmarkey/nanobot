"""Compare old (httpx) vs new (curl_cffi) fetch paths against a list of sites.

Runs both paths in parallel and classifies each response as:
  - OK      (2xx, not a CF challenge page)
  - CF      (Cloudflare challenge / interstitial)
  - BLOCK   (403/401/451 etc, non-CF — explicit bot block)
  - HTTP5XX (server error)
  - ERR     (network / timeout / TLS error)
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass

import httpx
from curl_cffi.requests import AsyncSession as CurlAsyncSession

SITES = sorted(set([
    # First block
    "amazon.com", "figma.com", "airtable.com", "trello.com", "dribbble.com",
    "huggingface.co", "coinmarketcap.com", "miro.com", "imdb.com", "booking.com",
    "hulu.com", "pbs.org", "wsj.com", "marketwatch.com", "jiosaavn.com",
    "mapquest.com", "justia.com", "missingkids.org", "siemens.com",
    "westernunion.com", "uber.com", "npr.org", "goodreads.com", "twilio.com",
    "realtor.com", "cloudconvert.com", "convertio.co", "note.com",
    "mercadolivre.com.br", "app.hubspot.com",
    # Second block
    "canva.com", "fiverr.com", "ahrefs.com", "gitlab.com", "discord.com",
    "perplexity.ai", "calendly.com", "chatgpt.com", "claude.ai", "grok.com",
    "accounts.shopify.com", "anydesk.com", "bbb.org", "aboutcookies.org",
    "allaboutcookies.org", "taringa.net", "techinasia.com", "techdirt.com",
    "teamliquid.net", "teespring.com", "bukkit.org", "bab.la", "petco.com",
    "cymax.com", "grgich.com", "mountcongreve.com", "huptechweb.com",
    "app.ahrefs.com",
]))

OLD_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36"
TIMEOUT = 15.0
CONCURRENCY = 8

CF_MARKERS = [
    "just a moment",
    "cf-browser-verification",
    "cf_chl_opt",
    "challenges.cloudflare.com",
    "checking if the site connection is secure",
    "please enable cookies",
    "enable javascript and cookies to continue",
]


@dataclass
class Result:
    kind: str       # OK | CF | BLOCK | HTTP5XX | ERR
    status: int     # 0 on ERR
    elapsed: float
    detail: str = ""


def classify(status: int, text: str, server: str) -> tuple[str, str]:
    head = (text or "")[:3000].lower()
    srv = (server or "").lower()

    is_cf_page = any(m in head for m in CF_MARKERS)
    if status in (403, 503) and "cloudflare" in srv and is_cf_page:
        return "CF", "cf-challenge"
    if is_cf_page and ("cloudflare" in srv or "challenges.cloudflare.com" in head):
        return "CF", "cf-challenge"
    if 200 <= status < 300:
        # Some CF interstitials return 200 (managed challenge)
        if is_cf_page:
            return "CF", "cf-interstitial-200"
        return "OK", f"{len(text)}B"
    if status in (401, 403, 451):
        return "BLOCK", f"http-{status}"
    if 500 <= status < 600:
        return "HTTP5XX", f"http-{status}"
    return "BLOCK", f"http-{status}"


async def fetch_old(url: str) -> Result:
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(
            follow_redirects=True, max_redirects=5, timeout=TIMEOUT,
        ) as client:
            r = await client.get(url, headers={"User-Agent": OLD_UA})
        kind, detail = classify(r.status_code, r.text, r.headers.get("server", ""))
        return Result(kind, r.status_code, time.perf_counter() - t0, detail)
    except Exception as e:
        return Result("ERR", 0, time.perf_counter() - t0, type(e).__name__)


async def fetch_new(url: str) -> Result:
    t0 = time.perf_counter()
    try:
        async with CurlAsyncSession(impersonate="chrome", timeout=TIMEOUT) as client:
            r = await client.get(url, allow_redirects=True, max_redirects=5)
        kind, detail = classify(r.status_code, r.text, r.headers.get("server", ""))
        return Result(kind, r.status_code, time.perf_counter() - t0, detail)
    except Exception as e:
        return Result("ERR", 0, time.perf_counter() - t0, type(e).__name__)


async def probe(sem: asyncio.Semaphore, host: str) -> tuple[str, Result, Result]:
    url = f"https://{host}/"
    async with sem:
        old, new = await asyncio.gather(fetch_old(url), fetch_new(url))
    return host, old, new


def badge(kind: str) -> str:
    return {"OK": "OK ", "CF": "CF ", "BLOCK": "BLK", "HTTP5XX": "5XX", "ERR": "ERR"}.get(kind, kind)


async def main() -> int:
    sem = asyncio.Semaphore(CONCURRENCY)
    results = await asyncio.gather(*(probe(sem, h) for h in SITES))

    # Sort: wins first (new succeeds where old failed), then unchanged, then regressions
    def rank(row):
        _, old, new = row
        if new.kind == "OK" and old.kind != "OK":
            return (0, row[0])
        if new.kind != "OK" and old.kind == "OK":
            return (2, row[0])
        return (1, row[0])

    results.sort(key=rank)

    header = f"{'host':<25} {'old':<30} {'new':<30} {'Δ'}"
    print(header)
    print("-" * len(header))

    counts = {"win": 0, "same_ok": 0, "same_fail": 0, "regress": 0}
    for host, old, new in results:
        old_s = f"{badge(old.kind)} {old.status or '-':>3} {old.detail[:18]:<18}"
        new_s = f"{badge(new.kind)} {new.status or '-':>3} {new.detail[:18]:<18}"
        if new.kind == "OK" and old.kind != "OK":
            delta = "WIN"
            counts["win"] += 1
        elif new.kind == "OK" and old.kind == "OK":
            delta = "="
            counts["same_ok"] += 1
        elif new.kind != "OK" and old.kind == "OK":
            delta = "REGRESS"
            counts["regress"] += 1
        else:
            delta = "both-fail"
            counts["same_fail"] += 1
        print(f"{host:<25} {old_s:<30} {new_s:<30} {delta}")

    print("-" * len(header))
    print(f"TOTAL: {len(results)}  WIN: {counts['win']}  ==: {counts['same_ok']}  "
          f"REGRESS: {counts['regress']}  both-fail: {counts['same_fail']}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
