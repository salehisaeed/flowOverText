"""
Minimal reader for OpenFOAM volume fields (ascii or binary, label 32-bit,
scalar 64-bit), as written by a decomposed or serial case.
"""

import os
import re

import numpy as np

_NCOMP = {"scalar": 1, "vector": 3, "sphericalTensor": 1,
          "symmTensor": 6, "tensor": 9}
_WS = b" \t\r\n"


class _Reader:
    def __init__(self, data, binary):
        self.d, self.i, self.binary = data, 0, binary

    def ws(self):
        d, i = self.d, self.i
        while i < len(d):
            c = d[i:i + 1]
            if c in (b" ", b"\t", b"\r", b"\n"):
                i += 1
            elif d.startswith(b"//", i):
                j = d.find(b"\n", i)
                i = len(d) if j < 0 else j + 1
            elif d.startswith(b"/*", i):
                j = d.find(b"*/", i)
                i = len(d) if j < 0 else j + 2
            else:
                break
        self.i = i

    def word(self):
        self.ws()
        m = re.compile(rb'"[^"]*"|[^\s;{}()]+').match(self.d, self.i)
        if not m:
            return None
        self.i = m.end()
        return m.group().strip(b'"').decode()

    def peek(self):
        self.ws()
        return self.d[self.i:self.i + 1]

    def expect(self, c):
        self.ws()
        if self.d[self.i:self.i + 1] != c:
            raise ValueError(f"expected {c!r} at byte {self.i}")
        self.i += 1

    def skip_entry(self):
        """Skip to the end of the current entry (';' or a balanced {...})."""
        depth = 0
        d = self.d
        while self.i < len(d):
            c = d[self.i:self.i + 1]
            if c == b"{":
                depth += 1
            elif c == b"}":
                if depth == 0:
                    return
                depth -= 1
                if depth == 0:
                    self.i += 1
                    return
            elif c == b";" and depth == 0:
                self.i += 1
                return
            self.i += 1

    def field_value(self, ncomp):
        """Parse 'uniform X' or 'nonuniform List<T> N (...)' (cursor after keyword)."""
        kind = self.word()
        if kind == "uniform":
            self.ws()
            j = self.d.find(b";", self.i)
            txt = self.d[self.i:j].replace(b"(", b" ").replace(b")", b" ")
            self.i = j + 1
            return ("uniform", np.array(txt.split(), float))
        typ = self.word()                       # List<scalar>
        comp = _NCOMP[typ[typ.index("<") + 1:-1]]
        self.ws()
        m = re.compile(rb"\d+").match(self.d, self.i)
        n = int(m.group())
        self.i = m.end()
        self.ws()
        if self.d[self.i:self.i + 1] == b";":   # empty list: 'List<T> 0;'
            self.i += 1
            return ("list", np.zeros((0, comp)) if comp > 1 else np.zeros(0))
        if self.d[self.i:self.i + 1] == b"{":   # N{value}
            self.i += 1
            j = self.d.find(b"}", self.i)
            v = np.array(self.d[self.i:j].replace(b"(", b" ").replace(b")", b" ").split(), float)
            self.i = j + 1
            self.skip_entry()
            return ("list", np.tile(v, (n, 1)) if comp > 1 else np.full(n, v[0]))
        self.expect(b"(")
        if self.binary and n > 0:
            nbytes = 8*n*comp
            arr = np.frombuffer(self.d, "<f8", n*comp, self.i).copy()
            self.i += nbytes
            self.expect(b")")
        elif n > 0:
            depth, j = 1, self.i
            while depth:
                c = self.d[j:j + 1]
                depth += (c == b"(") - (c == b")")
                j += 1
            txt = self.d[self.i:j - 1].replace(b"(", b" ").replace(b")", b" ")
            arr = np.array(txt.split(), float)
            self.i = j
        else:
            self.expect(b")")
            arr = np.zeros(0)
        self.skip_entry()
        return ("list", arr.reshape(n, comp) if comp > 1 else arr)


def read_field(path, patches=None):
    """Return (internal, {patch: values}) for a vol field file.

    Only patches with a 'value' entry (and listed in *patches*, if given)
    are returned.  Vectors come back as (N, 3) arrays.
    """
    with open(path, "rb") as f:
        data = f.read()
    hdr = data[:data.find(b"}") + 1]
    binary = re.search(rb"format\s+binary", hdr) is not None
    cls = re.search(rb"class\s+(\w+)", hdr).group(1).decode()
    ncomp = 3 if "Vector" in cls else 1

    r = _Reader(data, binary)
    r.i = data.find(b"}") + 1
    internal, boundary = None, {}
    while True:
        key = r.word()
        if key is None:
            break
        if key == "internalField":
            kind, v = r.field_value(ncomp)
            internal = v
        elif key == "boundaryField":
            r.expect(b"{")
            while r.peek() != b"}":
                name = r.word()
                r.expect(b"{")
                while r.peek() != b"}":
                    k = r.word()
                    r.ws()
                    if k == "value" and (patches is None or name in patches):
                        boundary[name] = r.field_value(ncomp)
                    elif r.d.startswith((b"uniform", b"nonuniform"), r.i):
                        r.field_value(ncomp)
                    else:
                        r.skip_entry()
                r.expect(b"}")
            r.expect(b"}")
        else:
            r.skip_entry()
    return internal, boundary


def processor_dirs(case):
    procs = sorted((d for d in os.listdir(case) if re.fullmatch(r"processor\d+", d)),
                   key=lambda s: int(s[9:]))
    return [os.path.join(case, p) for p in procs] or [case]


def time_dirs(root, field):
    """Sorted (value, name) of time directories in *root* containing *field*."""
    out = []
    for d in os.listdir(root):
        try:
            t = float(d)
        except ValueError:
            continue
        if os.path.isfile(os.path.join(root, d, field)):
            out.append((t, d))
    return sorted(out)
