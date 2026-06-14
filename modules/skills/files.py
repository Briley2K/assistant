"""
Read-only file-system access for Cleo.

These skills let the user ask things like "read my todo list" or "what's in my
Documents folder". Access is deliberately narrow and strictly read-only:

  • Cleo can only see paths inside the allowed roots (config.FILE_ACCESS_ROOTS),
    granted from the control panel. With no roots set, nothing is readable.
  • Anything matching the blacklist (config.FILE_ACCESS_BLACKLIST) is invisible —
    hidden from listings/searches and refused on read, even inside an allowed
    root (so e.g. a .env under an allowed project folder stays secret).
  • Reads are size-capped (config.FILE_ACCESS_MAX_KB) and binary files are refused.

Every path is resolved with realpath BEFORE the checks, so a symlink can't escape
an allowed root or point at a blacklisted target.
"""
import os
import fnmatch
import subprocess

import config
from modules.skills import skill


# Spoken folder name → XDG user-dir key (resolved via `xdg-user-dir`, which
# respects ~/.config/user-dirs.dirs, so a relocated "Downloads" still resolves).
_XDG_DIRS = {
    "desktop": "DESKTOP",
    "download": "DOWNLOAD", "downloads": "DOWNLOAD",
    "document": "DOCUMENTS", "documents": "DOCUMENTS", "docs": "DOCUMENTS",
    "music": "MUSIC",
    "picture": "PICTURES", "pictures": "PICTURES", "photos": "PICTURES",
    "video": "VIDEOS", "videos": "VIDEOS", "movies": "VIDEOS",
    "public": "PUBLICSHARE", "templates": "TEMPLATES",
}


def _home() -> str:
    return os.path.expanduser("~")


def _xdg_dir(key: str) -> str | None:
    """Resolve an XDG user-dir key (e.g. DOWNLOAD) to its real path, or None."""
    try:
        r = subprocess.run(["xdg-user-dir", key], capture_output=True,
                            text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    path = r.stdout.strip().rstrip("/")
    # xdg-user-dir echoes $HOME (sometimes with a trailing slash) when a key is
    # unset — compare resolved paths so that case counts as "no match".
    if path and os.path.realpath(path) != os.path.realpath(_home()) and os.path.isdir(path):
        return path
    return None


def _norm_dir_name(name: str) -> str:
    """Strip filler words from a spoken folder name: 'my downloads folder' → 'downloads'."""
    words = [w for w in str(name).lower().split()
             if w not in ("my", "the", "folder", "directory", "dir")]
    return " ".join(words).strip()


def known_locations() -> dict:
    """Common user folders that actually exist AND fall inside an allowed root:
    {label: real_path}. Used both to answer find_directory and to tell the model
    these paths up front (so it usually doesn't need to search at all)."""
    roots = [_real(r) for r in config.FILE_ACCESS_ROOTS]
    if not roots:
        return {}
    out = {}
    candidates = [("Home", _home())]
    seen_keys = set()
    for spoken, key in _XDG_DIRS.items():
        if key in seen_keys:
            continue
        seen_keys.add(key)
        path = _xdg_dir(key) or os.path.join(_home(), spoken.capitalize())
        candidates.append((key.capitalize(), path))
    for label, path in candidates:
        rp = _real(path)
        if os.path.isdir(rp) and any(_under(rp, r) for r in roots) and not _blacklisted(rp):
            out[label] = rp
    return out


def _real(path: str) -> str:
    return os.path.realpath(os.path.expanduser(str(path)))


def _under(path: str, root: str) -> bool:
    """True if `path` is `root` itself or lives inside it (both realpath'd)."""
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _blacklisted(real_path: str) -> bool:
    """True if a resolved path matches any blacklist entry — a glob (matched
    against the full path and the bare name) or a literal path / parent folder."""
    base = os.path.basename(real_path)
    for entry in config.FILE_ACCESS_BLACKLIST:
        if any(c in entry for c in "*?["):
            if fnmatch.fnmatch(real_path, entry) or fnmatch.fnmatch(base, entry):
                return True
        else:
            er = _real(entry)
            if _under(real_path, er):
                return True
    return False


def _check(path):
    """Resolve `path` and decide whether Cleo may touch it. Returns
    (real_path, None) when allowed, or (None, error_dict) when not."""
    if not config.FILE_ACCESS_ENABLED:
        return None, {"error": "file access is turned off in the control panel"}
    if not config.FILE_ACCESS_ROOTS:
        return None, {"error": "no folders have been shared with me yet — add one in the control panel"}
    if not path or not str(path).strip():
        return None, {"error": "no path given"}
    real = _real(path)
    if not any(_under(real, _real(r)) for r in config.FILE_ACCESS_ROOTS):
        return None, {"error": f"'{path}' is outside the folders I'm allowed to read"}
    if _blacklisted(real):
        return None, {"error": f"'{path}' is on the blocked list"}
    return real, None


def locations_hint() -> str:
    """A directive for the system prompt: tells the model it CAN read the user's
    files, that it must use the file tools to do so (rather than refuse or guess),
    and the real absolute paths of the user's common folders. Empty when file
    access is off / nothing is shared."""
    if not config.FILE_ACCESS_ENABLED:
        return ""
    locs = known_locations()
    if not locs:
        return ""
    pairs = ", ".join(f"{label} is {path}" for label, path in locs.items())
    return (
        "You HAVE read-only access to the user's files on this computer. Whenever "
        "the user asks what files or folders they have, or to read a file, you MUST "
        "answer by calling a file tool — list_directory to see a folder, read_file "
        "to read a file, find_directory to locate a folder by name. NEVER say you "
        "can't access their files, and never guess at the contents — call the tool "
        "and use its result. Only if a tool itself returns an error do you tell the "
        "user there was a problem. The user's folders are at these absolute paths: "
        f"{pairs}. Use these paths directly (e.g. list_directory on "
        f"{locs.get('Download', next(iter(locs.values())))} for their downloads). "
        "For any other folder, call find_directory with its name first.")


@skill("list_directory",
       "List the files and sub-folders in a directory you have access to.",
       {"path": "absolute path of the folder to list"})
def _list_directory(args):
    given = args.get("path") or args.get("dir") or args.get("folder") or ""
    real, err = _check(given)
    if err:
        return err
    if not os.path.isdir(real):
        return {"error": f"not a folder: {given}"}
    try:
        names = sorted(os.listdir(real))
    except OSError as e:
        return {"error": f"couldn't read folder: {e}"}
    entries, hidden = [], 0
    for name in names:
        full = os.path.join(real, name)
        if _blacklisted(_real(full)):
            hidden += 1                       # don't even reveal blocked items
            continue
        try:
            is_dir = os.path.isdir(full)
            size = None if is_dir else os.path.getsize(full)
        except OSError:
            continue
        entries.append({"name": name, "type": "dir" if is_dir else "file",
                        **({"size_kb": round(size / 1024, 1)} if size is not None else {})})
        if len(entries) >= 200:
            break
    out = {"path": real, "count": len(entries), "entries": entries}
    if hidden:
        out["hidden"] = hidden                # count of blacklisted items skipped
    return out


@skill("read_file",
       "Read the text contents of a file you have access to.",
       {"path": "absolute path of the file to read"})
def _read_file(args):
    given = args.get("path") or args.get("file") or ""
    real, err = _check(given)
    if err:
        return err
    if not os.path.isfile(real):
        return {"error": f"not a file: {given}"}
    cap = config.FILE_ACCESS_MAX_KB * 1024
    try:
        size = os.path.getsize(real)
        with open(real, "rb") as fh:
            raw = fh.read(cap + 1)
    except OSError as e:
        return {"error": f"couldn't read file: {e}"}
    if b"\x00" in raw:
        return {"error": "that looks like a binary file, not text"}
    truncated = len(raw) > cap
    text = raw[:cap].decode("utf-8", "replace")
    return {"path": real, "size_kb": round(size / 1024, 1),
            "truncated": truncated, "content": text}


@skill("search_files",
       "Find files by name within the folders you can access.",
       {"query": "part of a file name to search for",
        "path": "optional folder to search under (defaults to every allowed folder)"})
def _search_files(args):
    query = (args.get("query") or args.get("name") or "").strip().lower()
    if not query:
        return {"error": "no search text given"}
    if not config.FILE_ACCESS_ENABLED:
        return {"error": "file access is turned off in the control panel"}
    roots = config.FILE_ACCESS_ROOTS
    if args.get("path"):
        real, err = _check(args.get("path"))
        if err:
            return err
        roots = [real]
    if not roots:
        return {"error": "no folders have been shared with me yet"}
    hits = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(_real(root)):
            # Prune blacklisted sub-folders so we never descend into them.
            dirnames[:] = [d for d in dirnames
                           if not _blacklisted(_real(os.path.join(dirpath, d)))]
            for fn in filenames:
                if query not in fn.lower():
                    continue
                full = _real(os.path.join(dirpath, fn))
                if _blacklisted(full):
                    continue
                hits.append(full)
                if len(hits) >= 100:
                    return {"query": query, "count": len(hits),
                            "matches": hits, "truncated": True}
    return {"query": query, "count": len(hits), "matches": hits}


# Directory names never worth descending into when hunting for a folder — caches,
# version-control internals, and package trees that would swamp the walk.
_SKIP_WALK_DIRS = {".git", ".cache", "node_modules", "__pycache__",
                   ".venv", "venv", "site-packages", ".local", "snap"}


@skill("find_directory",
       "Find a folder by name (e.g. Downloads, Desktop, a project folder) and "
       "return its full path. Use this to locate a folder before listing it.",
       {"name": "the folder name to find, e.g. 'Downloads' or 'Desktop'"})
def _find_directory(args):
    if not config.FILE_ACCESS_ENABLED:
        return {"error": "file access is turned off in the control panel"}
    if not config.FILE_ACCESS_ROOTS:
        return {"error": "no folders have been shared with me yet"}
    raw = (args.get("name") or args.get("query") or args.get("path") or "").strip()
    name = _norm_dir_name(raw)
    if not name:
        return {"error": "no folder name given"}

    roots = [_real(r) for r in config.FILE_ACCESS_ROOTS]

    def _ok(rp):
        return any(_under(rp, r) for r in roots) and not _blacklisted(rp)

    # 1) Fast path: a well-known folder (XDG user-dir or ~/<Name>).
    key = _XDG_DIRS.get(name)
    if key:
        path = _xdg_dir(key) or os.path.join(_home(), name.capitalize())
        rp = _real(path)
        if os.path.isdir(rp) and _ok(rp):
            return {"name": raw, "count": 1, "matches": [rp], "resolved": "known"}
    direct = _real(os.path.join(_home(), raw))
    if os.path.isdir(direct) and _ok(direct):
        return {"name": raw, "count": 1, "matches": [direct], "resolved": "home"}

    # 2) Walk: home first (most likely), then the rest of the allowed roots.
    needle = name.lower()
    search_roots, seen = [], set()
    for r in [_real(_home())] + roots:
        if os.path.isdir(r) and r not in seen and any(_under(r, x) for x in roots):
            seen.add(r); search_roots.append(r)
    # Gather exact (case-insensitive) name matches separately from substring
    # matches; exact wins. Searching home first means home hits rank above the
    # rest of the filesystem. Stop early once we have enough exact matches.
    exact, partial = [], []
    for root in search_roots:
        for dirpath, dirnames, _ in os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if d not in _SKIP_WALK_DIRS
                           and not _blacklisted(_real(os.path.join(dirpath, d)))]
            for d in dirnames:
                low = d.lower()
                if needle not in low:
                    continue
                rp = _real(os.path.join(dirpath, d))
                if not _ok(rp):
                    continue
                if low == needle and rp not in exact:
                    exact.append(rp)
                elif rp not in partial:
                    partial.append(rp)
            if len(exact) >= 10:
                break
        if len(exact) >= 10:
            break
    hits = exact + [p for p in partial if p not in exact]
    if not hits:
        return {"name": raw, "count": 0, "matches": [],
                "note": "no folder by that name found in the folders I can access"}
    hits = hits[:50]
    return {"name": raw, "count": len(hits), "matches": hits,
            "truncated": len(exact) + len(partial) > len(hits)}
