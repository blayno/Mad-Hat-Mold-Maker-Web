"""Offline support: serves three.js and the UI font from static/vendor.

If the files have not been downloaded yet, the routes redirect to the CDN, so
the app still works online exactly as before.  Run `python vendor.py` once
while online (or just start the app; it tries in the background) and from then
on everything loads from this computer.
"""
from __future__ import annotations

import hashlib
import re
import threading
import urllib.request
from pathlib import Path

from flask import abort, redirect, send_from_directory

THREE_VERSION = '0.160.0'
CDN = f'https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/'
THREE_FILES = (
    'build/three.module.js',
    'examples/jsm/controls/OrbitControls.js',
    'examples/jsm/loaders/STLLoader.js',
    'examples/jsm/utils/BufferGeometryUtils.js',
)
FONT_CSS_URL = ('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque'
                ':opsz,wght@12..96,600;12..96,700&display=swap')
ROOT = Path(__file__).resolve().parent / 'static' / 'vendor'
_BROWSER = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
            '(KHTML, like Gecko) Chrome/124.0 Safari/537.36')   # makes Google serve woff2
_FONT_FILE = re.compile(r'^[A-Za-z0-9_.-]+\.woff2$')


def register(app):
    @app.get('/vendor/three/<path:name>')
    def vendor_three(name):
        if name not in THREE_FILES:
            abort(404)
        if (ROOT / 'three' / name).is_file():
            return send_from_directory(ROOT / 'three', name, mimetype='text/javascript', max_age=86400)
        return redirect(CDN + name, 302)

    @app.get('/vendor/fonts.css')
    def vendor_fonts_css():
        if (ROOT / 'fonts' / 'fonts.css').is_file():
            return send_from_directory(ROOT / 'fonts', 'fonts.css', mimetype='text/css', max_age=86400)
        return redirect(FONT_CSS_URL, 302)

    @app.get('/vendor/fonts/<name>')
    def vendor_font_file(name):
        if not _FONT_FILE.match(name) or not (ROOT / 'fonts' / name).is_file():
            abort(404)
        return send_from_directory(ROOT / 'fonts', name, mimetype='font/woff2', max_age=86400)


def _get(url, timeout=25):
    req = urllib.request.Request(url, headers={'User-Agent': _BROWSER})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_bytes(data)
    tmp.replace(path)                      # never leaves a half-written file behind


def missing():
    out = [f for f in THREE_FILES if not (ROOT / 'three' / f).is_file()]
    if not (ROOT / 'fonts' / 'fonts.css').is_file():
        out.append('fonts')
    return out


def fetch(log=print):
    """Download everything that is missing.  Returns True when nothing is left."""
    for f in [m for m in missing() if m != 'fonts']:
        log(f'Downloading {f} …')
        _write(ROOT / 'three' / f, _get(CDN + f))
    if 'fonts' in missing():
        log('Downloading the UI font …')
        css = _get(FONT_CSS_URL).decode('utf-8')
        for url in sorted(set(re.findall(r'url\((https://fonts\.gstatic\.com/[^)]+)\)', css))):
            local = hashlib.sha1(url.encode()).hexdigest()[:12] + '.woff2'
            if not (ROOT / 'fonts' / local).is_file():
                _write(ROOT / 'fonts' / local, _get(url))
            css = css.replace(url, '/vendor/fonts/' + local)
        _write(ROOT / 'fonts' / 'fonts.css', css.encode('utf-8'))   # last, so it only exists when complete
    return not missing()


def ensure_async():
    """At app start: fetch missing files in the background; stay silent if offline."""
    if not missing():
        return

    def work():
        try:
            fetch(log=lambda m: None)
        except Exception:
            pass

    threading.Thread(target=work, daemon=True).start()


if __name__ == '__main__':
    try:
        ok = fetch()
    except Exception as e:
        raise SystemExit(f'Could not download: {e}\nConnect to the internet and run this again.')
    print('Done. The app now runs fully offline.' if ok else 'Some files are still missing.')
