"""
STEP 0 (only if you do not already have the histopathology images)

Downloads ONLY the histopathology images from the DOLCHID archive on figshare,
without downloading the CBCT volumes again. It reads the ZIP central directory
with HTTP range requests (ZIP64 supported) and extracts the matching entries.

    python step0_download_histology.py --list      # show what is in the archive
    python step0_download_histology.py             # download histology images

If you already have the PNGs (e.g. from the earlier Kaggle run), just copy them
into config.HIST_DIR and skip this step.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # embedded-style Python (._pth) does not add the script folder
import argparse
import re
import struct
import time
import zlib
from pathlib import Path

import config as C
from common import log

IMG_EXT = re.compile(r"\.(png|jpe?g|tiff?|bmp)$", re.IGNORECASE)
ID_IN_NAME = re.compile(r"(?<![A-Za-z])(AME|DC|KCOT|RC)[ _-]?(\d+)", re.IGNORECASE)
API = "https://api.figshare.com/v2/articles/{aid}/files?page_size=1000"
ARTICLE_ID = 30156622  # https://doi.org/10.6084/m9.figshare.30156622


def is_hist(path):
    """Histology image: an image file whose path mentions histo/patho/HE/HIST."""
    if not IMG_EXT.search(path):
        return False
    low = path.lower()
    return any(k in low for k in ("hist", "patho", "_he", "/he", "h&e", "slide"))


def target_name(path):
    m = ID_IN_NAME.search(path.split("/")[-1]) or ID_IN_NAME.search(path)
    if not m:
        return None
    ext = IMG_EXT.search(path).group(0).lower()
    return f"{m.group(1).upper()}_{int(m.group(2))}_HIST_Image{ext}"


class RemoteFile:
    def __init__(self, url, retries=5, timeout=300):
        import requests
        self.orig_url = url
        self.url, self.retries, self.timeout = url, retries, timeout
        self.s = requests.Session()
        r = self.s.head(url, allow_redirects=True, timeout=timeout)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])
        self.url = r.url  # resolved (signed) URL

    def read(self, start, length):
        end = start + length - 1
        for attempt in range(1, self.retries + 1):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={start}-{end}"},
                               timeout=self.timeout)
                if r.status_code in (403, 400):  # signed URL may expire -> re-resolve
                    self.__init__(self.orig_url, self.retries, self.timeout)
                    raise RuntimeError(f"HTTP {r.status_code}, URL refreshed")
                if r.status_code != 206:
                    raise RuntimeError(f"Server did not honour range request (HTTP {r.status_code})")
                if len(r.content) != length:
                    raise RuntimeError("Incomplete range read")
                return r.content
            except Exception as e:
                log(f"  retry {attempt}/{self.retries}: {e}")
                if attempt == self.retries:
                    raise
                time.sleep(3 * attempt)


class LocalFile:
    """Same interface for a local zip (used by the self-test)."""
    def __init__(self, path):
        self.f = open(path, "rb")
        self.f.seek(0, 2)
        self.size = self.f.tell()

    def read(self, start, length):
        self.f.seek(start)
        return self.f.read(length)


def central_directory(src):
    tail_len = min(src.size, 65536 + 22)
    tail = src.read(src.size - tail_len, tail_len)
    pos = tail.rfind(b"PK\x05\x06")
    if pos < 0:
        raise RuntimeError("End-of-central-directory record not found")
    (_, _, _, _, n_entries, cd_size, cd_offset, _) = struct.unpack("<IHHHHIIH", tail[pos:pos + 22])
    if cd_offset == 0xFFFFFFFF or n_entries == 0xFFFF or cd_size == 0xFFFFFFFF:
        loc = tail.rfind(b"PK\x06\x07", 0, pos)
        if loc < 0:
            raise RuntimeError("ZIP64 locator not found")
        eocd64_off = struct.unpack("<Q", tail[loc + 8:loc + 16])[0]
        rec = src.read(eocd64_off, 56)
        n_entries, cd_size, cd_offset = struct.unpack("<QQQ", rec[32:56])
    cd = src.read(cd_offset, cd_size)
    entries, p = [], 0
    for _ in range(n_entries):
        if cd[p:p + 4] != b"PK\x01\x02":
            raise RuntimeError("Bad central directory entry")
        (method, csize, usize, nlen, xlen, clen, lho) = (
            struct.unpack("<H", cd[p + 10:p + 12])[0],
            *struct.unpack("<II", cd[p + 20:p + 28]),
            *struct.unpack("<HHH", cd[p + 28:p + 34]),
            struct.unpack("<I", cd[p + 42:p + 46])[0])
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
        if 0xFFFFFFFF in (csize, usize, lho):
            q = 0
            while q + 4 <= len(extra):
                hid, hlen = struct.unpack("<HH", extra[q:q + 4])
                if hid == 0x0001:
                    vals, k = extra[q + 4:q + 4 + hlen], 0
                    if usize == 0xFFFFFFFF:
                        usize = struct.unpack("<Q", vals[k:k + 8])[0]; k += 8
                    if csize == 0xFFFFFFFF:
                        csize = struct.unpack("<Q", vals[k:k + 8])[0]; k += 8
                    if lho == 0xFFFFFFFF:
                        lho = struct.unpack("<Q", vals[k:k + 8])[0]; k += 8
                q += 4 + hlen
        entries.append(dict(name=name, method=method, csize=csize, usize=usize, offset=lho))
        p += 46 + nlen + xlen + clen
    return entries


def extract_entry(src, e):
    hdr = src.read(e["offset"], 30)
    if hdr[:4] != b"PK\x03\x04":
        raise RuntimeError("Bad local header")
    nlen, xlen = struct.unpack("<HH", hdr[26:30])
    data = src.read(e["offset"] + 30 + nlen + xlen, e["csize"])
    if e["method"] == 0:
        out = data
    elif e["method"] == 8:
        out = zlib.decompress(data, -zlib.MAX_WBITS)
    else:
        raise RuntimeError(f"Unsupported compression method {e['method']}")
    if len(out) != e["usize"]:
        raise RuntimeError("Size mismatch after decompression")
    return out


def figshare_files():
    import requests
    r = requests.get(API.format(aid=ARTICLE_ID), timeout=60)
    r.raise_for_status()
    return [dict(name=f["name"], size=f["size"], url=f["download_url"]) for f in r.json()]


def save(dst, data, log_name):
    dst.write_bytes(data)
    log(f"  saved {dst.name}  <- {log_name}")


def harvest_zip(src, label, list_only):
    entries = central_directory(src)
    hist = [e for e in entries if is_hist(e["name"])]
    log(f"{label}: {len(entries)} entries, {len(hist)} histology images "
        f"({sum(e['csize'] for e in hist) / 1e9:.2f} GB compressed)")
    if list_only:
        for e in entries[:40]:
            print(f"{e['csize'] / 1e6:10.2f} MB  {e['name']}")
        if hist:
            print("  example histology entries:", [e["name"] for e in hist[:5]])
        return 0
    n = 0
    for i, e in enumerate(hist, 1):
        name = target_name(e["name"])
        if name is None:
            log(f"  skip (no patient ID in name): {e['name']}")
            continue
        dst = C.HIST_DIR / name
        if dst.exists() and dst.stat().st_size == e["usize"]:
            n += 1
            continue
        log(f"[{i}/{len(hist)}] {e['name']} ({e['csize'] / 1e6:.1f} MB)")
        save(dst, extract_entry(src, e), e["name"])
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list contents only, download nothing")
    ap.add_argument("--url", default=None, help="direct URL of one zip file (optional)")
    ap.add_argument("--local-zip", default=None, help="use a local zip instead")
    args = ap.parse_args()
    C.HIST_DIR.mkdir(parents=True, exist_ok=True)

    if args.local_zip or args.url:
        src = LocalFile(args.local_zip) if args.local_zip else RemoteFile(args.url)
        n = harvest_zip(src, args.local_zip or args.url, args.list)
    else:
        import requests
        files = figshare_files()
        log(f"figshare article {ARTICLE_ID}: {len(files)} files")
        for f in files:
            print(f"   {f['size'] / 1e9:8.2f} GB  {f['name']}")
        n = 0
        for f in files:
            if is_hist(f["name"]) or (IMG_EXT.search(f["name"]) and target_name(f["name"])):
                if args.list:
                    continue
                name = target_name(f["name"])
                dst = C.HIST_DIR / name
                if not (dst.exists() and dst.stat().st_size == f["size"]):
                    save(dst, requests.get(f["url"], timeout=600).content, f["name"])
                n += 1
            elif f["name"].lower().endswith(".zip"):
                n += harvest_zip(RemoteFile(f["url"]), f["name"], args.list)
    if not args.list:
        log(f"Done: {n} histopathology images in {C.HIST_DIR}")
        if n == 0:
            log("No histology images found automatically. Copy them manually into the folder above.")


if __name__ == "__main__":
    main()
