"""Build the optional C extensions: MSVC on Windows, gcc on Linux.

    python -B native/build.py                 # all modules of this platform
    python -B native/build.py _pipemux        # one of them

* ``spire._native`` (native/spire_native.c): batch tokenizer + collation kernels. Without it
  ``spire.tokenize`` uses the numpy batch encoder, which produces identical output
  (tests/test_tokenize_batch.py checks both paths against the frozen reference).
* ``spire._pipemux`` (native/pipemux.c): the overlapped named-pipe I/O of ``spire.pipes.PipeMux``.
  Without it the pure-Python implementation is used (tests/test_pipes.py runs both).

Windows: the compiler is located with vswhere; set ``VCVARS`` to a ``vcvars64.bat`` to override.
Object files go to ``native/build/``. Linux: ``gcc`` (override with ``CC``) builds ``_native`` only;
``_pipemux`` drives Windows named pipes and does not exist there. A module that is loaded by a running process cannot be
overwritten on Windows: stop such processes first.
"""
from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / "spire"
MODULES = {"_native": (HERE / "spire_native.c", "", True), "_pipemux": (HERE / "pipemux.c", "kernel32.lib", False)}
"""name -> (source, extra libraries, needs the numpy headers)"""
LINUX_MODULES = ("_native",)
VSWHERE = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
    "Microsoft Visual Studio" / "Installer" / "vswhere.exe"


def find_vcvars() -> Path:
    env = os.environ.get("VCVARS")
    if env:
        return Path(env)
    if VSWHERE.exists():
        out = subprocess.run([str(VSWHERE), "-latest", "-products", "*", "-requires",
                              "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
                             capture_output=True, text=True, check=False).stdout.strip()
        for line in out.splitlines():
            p = Path(line.strip()) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
            if p.exists():
                return p
    raise SystemExit("MSVC not found (install the C++ build tools or set VCVARS=...\\vcvars64.bat)")


def build(name: str, vcvars: Path) -> int:
    src, libs_extra, numpy_headers = MODULES[name]
    base = Path(sys.base_prefix)
    inc, libs = base / "include", base / "libs"
    if numpy_headers:
        import numpy
        libs_extra = f'/I"{numpy.get_include()}" {libs_extra}'.strip()
    out = PKG / f"{name}{sysconfig.get_config_var('EXT_SUFFIX')}"
    build_dir = HERE / "build"
    build_dir.mkdir(exist_ok=True)
    bat = build_dir / f"build{name}.bat"
    bat.write_text(
        "@echo off\r\n"
        f'call "{vcvars}" >nul || exit /b 1\r\n'
        f'cl /nologo /O2 /W3 /LD /MD /Fo"{build_dir}\\\\" /I"{inc}" "{src}" {libs_extra} '
        f'/link /NOLOGO /LIBPATH:"{libs}" /OUT:"{out}" /IMPLIB:"{build_dir / (name + ".lib")}"\r\n',
        encoding="utf-8")
    r = subprocess.run(["cmd", "/c", str(bat)], capture_output=True, text=True, check=False)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0 or not out.exists():
        print(f"build of {name} failed (exit {r.returncode})", file=sys.stderr)
        return 1
    for junk in (out.with_suffix(".exp"), out.with_suffix(".lib")):
        if junk.exists():
            junk.unlink()
    print(f"built {out}")
    return 0


def build_gcc(name: str) -> int:
    """gcc counterpart of ``build``: the same -O2 level as MSVC's /O2, no -ffast-math, so the
    float results of the kernel match the MSVC build (checked by bench/remote identity hashes)."""
    src, _, numpy_headers = MODULES[name]
    out = PKG / f"{name}{sysconfig.get_config_var('EXT_SUFFIX')}"
    cmd = [os.environ.get("CC", "gcc"), "-O2", "-Wall", "-shared", "-fPIC", "-fno-strict-aliasing",
           f"-I{sysconfig.get_paths()['include']}"]
    if numpy_headers:
        import numpy
        cmd.append(f"-I{numpy.get_include()}")
    cmd += [str(src), "-o", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0 or not out.exists():
        print(f"build of {name} failed (exit {r.returncode})", file=sys.stderr)
        return 1
    print(f"built {out}")
    return 0


def main() -> int:
    if sys.platform == "linux":
        names = sys.argv[1:] or list(LINUX_MODULES)
        unknown = [n for n in names if n not in LINUX_MODULES]
        if unknown:
            raise SystemExit(f"module(s) {unknown} are not built on Linux; choose from {list(LINUX_MODULES)}")
        return max(build_gcc(n) for n in names)
    if sys.platform != "win32":
        raise SystemExit(f"no build recipe for {sys.platform}; compile the native/*.c files with your C compiler")
    names = sys.argv[1:] or list(MODULES)
    unknown = [n for n in names if n not in MODULES]
    if unknown:
        raise SystemExit(f"unknown module(s) {unknown}; choose from {list(MODULES)}")
    vcvars = find_vcvars()
    return max(build(n, vcvars) for n in names)


if __name__ == "__main__":
    sys.exit(main())
