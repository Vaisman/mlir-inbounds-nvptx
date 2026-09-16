#!/usr/bin/env python3
"""Drop the `gpu.module` wrapper so mlir-translate sees a plain LLVM-dialect module."""
import sys

lines = sys.stdin.read().split("\n")
start = next(i for i, l in enumerate(lines) if "gpu.module" in l)
indent = len(lines[start]) - len(lines[start].lstrip())
# Find the closing brace at the same indentation level.
end = next(i for i in range(start + 1, len(lines))
           if lines[i].strip() == "}" and (len(lines[i]) - len(lines[i].lstrip())) == indent)
out = lines[:start] + lines[start + 1:end] + lines[end + 1:]
sys.stdout.write("\n".join(out))
