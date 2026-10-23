"""Build script.

The Cython kernel is optional: without Cython or a C compiler the package
installs with the pure-NumPy kernel. OpenMP is optional too: the kernel is
first built with OpenMP and, if the compiler cannot do that (e.g. Apple
clang without libomp), rebuilt single-threaded.

Environment switches:
  ROBOLAB_NO_CYTHON=1   pure-NumPy install
  ROBOLAB_NO_OPENMP=1   single-threaded kernel
  ROBOLAB_NATIVE=1      tune for this CPU (-mcpu=native on ARM, -march=native on x86);
                        the binary then only runs on the same CPU model
"""
import os
import platform

from setuptools import setup
from setuptools.command.build_ext import build_ext


def _openmp_flags(compiler_type):  # noqa: E302
    if compiler_type == "msvc":
        return ["/openmp"], []
    return ["-fopenmp"], ["-fopenmp"]


def _native_flags(compiler_type):
    if os.environ.get("ROBOLAB_NATIVE") != "1" or compiler_type == "msvc":
        return []
    arm = platform.machine().lower() in ("aarch64", "arm64", "armv7l", "armv8l")
    return ["-mcpu=native"] if arm else ["-march=native"]


class BuildExt(build_ext):
    """Try the fastest configuration first and fall back step by step:
    native+OpenMP -> OpenMP -> native -> plain. Only if all fail does the
    package install without the compiled kernel (pure NumPy)."""

    def build_extension(self, ext):
        ct = self.compiler.compiler_type
        native = _native_flags(ct)
        omp_c, omp_l = ([], []) if os.environ.get("ROBOLAB_NO_OPENMP") == "1" else _openmp_flags(ct)
        base_c, base_l = list(ext.extra_compile_args), list(ext.extra_link_args)
        attempts = []
        for use_native in ([True, False] if native else [False]):
            for use_omp in ([True, False] if omp_c else [False]):
                attempts.append((use_native, use_omp))
        attempts.sort(key=lambda a: (not a[1], not a[0]))  # OpenMP matters more than native
        last = None
        for use_native, use_omp in attempts:
            ext.extra_compile_args = base_c + (native if use_native else []) + (omp_c if use_omp else [])
            ext.extra_link_args = base_l + (omp_l if use_omp else [])
            try:
                super().build_extension(ext)
            except Exception as exc:  # compiler rejected a flag
                last = exc
                print(f"robolab: build with native={use_native} openmp={use_omp} failed: {exc}")
                continue
            print(f"robolab: Cython kernel built, OpenMP={'yes' if use_omp else 'no'}, "
                  f"CPU-specific flags={native if use_native else 'none'}")
            return
        raise last


ext_modules = []
if os.environ.get("ROBOLAB_NO_CYTHON") != "1":
    try:
        import numpy as np
        from Cython.Build import cythonize
        from setuptools import Extension

        ext = Extension(
            "robolab.transforms._core_cy",
            ["robolab/transforms/_core_cy.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3"] if os.name != "nt" else ["/O2"],
            optional=True,  # a failed compile is a warning, not an error
        )
        ext_modules = cythonize([ext], language_level=3)
    except ImportError:
        pass

setup(ext_modules=ext_modules, cmdclass={"build_ext": BuildExt})
