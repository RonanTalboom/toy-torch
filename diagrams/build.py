#!/usr/bin/env python3
"""
Build Excalidraw diagrams for the toy-torch implementation milestones.

Writes raw .excalidraw JSON (same format as existing Cider brewing process.excalidraw)
into the vault's `toy-torch diagrams/` folder. Each diagram references actual
repo paths via text labels.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

OUT_DIR = "/Users/ronan/Library/Mobile Documents/iCloud~md~obsidian/Documents/Ronan /toy-torch diagrams"

PALETTE = {
    "stroke": "#1e1e1e",
    "text": "#1e1e1e",
    "title": "#1e1e1e",
    "input": "#b2d8b2",     # soft green — inputs / leaves
    "transform": "#d8cdf0",  # soft purple — transforms
    "output": "#f0d8a3",     # soft orange — outputs / artifacts
    "compiler": "#a3d0e0",   # soft blue — compiler stages
    "highlight": "#ffe4a0",  # bright yellow — headline results
    "bad": "#f5b5b5",        # soft red — slow / avoided
    "good": "#95d3a0",       # firm green — fast / preferred
    "transparent": "transparent",
}


class Builder:
    def __init__(self) -> None:
        self.elements: list[dict[str, Any]] = []
        self.counter = 1000

    def _id(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}{self.counter}"

    def title(self, text: str, x: int, y: int, width: int = 700) -> str:
        i = self._id("t")
        self.elements.append(_text(i, x, y, width, 48, text, font_size=32, align="center"))
        return i

    def subtitle(self, text: str, x: int, y: int, width: int = 700) -> str:
        i = self._id("st")
        self.elements.append(_text(i, x, y, width, 26, text, font_size=16, align="center"))
        return i

    def box(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        label: str,
        subtitle: Optional[str] = None,
        color: str = PALETTE["input"],
    ) -> str:
        bid = self._id("b")
        rect = {
            **_base(bid, "rectangle", x, y, w, h),
            "backgroundColor": color,
            "fillStyle": "solid",
            "strokeWidth": 2,
            "roundness": {"type": 3},
        }
        self.elements.append(rect)
        lbl = self._id("bl")
        # vertical centering heuristic
        if subtitle:
            self.elements.append(
                _text(lbl, x + 8, y + 10, w - 16, 22, label, font_size=18, align="center")
            )
            sub = self._id("bs")
            self.elements.append(
                _text(sub, x + 8, y + h - 30, w - 16, 20, subtitle, font_size=12, align="center")
            )
        else:
            self.elements.append(
                _text(lbl, x + 8, y + (h - 26) // 2, w - 16, 26, label, font_size=16, align="center")
            )
        return bid

    def diamond(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        label: str,
        color: str = PALETTE["transform"],
    ) -> str:
        did = self._id("d")
        rect = {
            **_base(did, "diamond", x, y, w, h),
            "backgroundColor": color,
            "fillStyle": "solid",
            "strokeWidth": 2,
        }
        self.elements.append(rect)
        lbl = self._id("dl")
        self.elements.append(
            _text(lbl, x + 8, y + (h - 24) // 2, w - 16, 24, label, font_size=14, align="center")
        )
        return did

    def arrow(
        self,
        from_id: str,
        to_id: str,
        start_xy: tuple[int, int],
        end_xy: tuple[int, int],
        label: Optional[str] = None,
    ) -> str:
        aid = self._id("a")
        sx, sy = start_xy
        ex, ey = end_xy
        self.elements.append(
            {
                **_base(aid, "arrow", sx, sy, ex - sx, ey - sy),
                "strokeWidth": 2,
                "points": [[0, 0], [ex - sx, ey - sy]],
                "lastCommittedPoint": [ex - sx, ey - sy],
                "startBinding": {"elementId": from_id, "focus": 0, "gap": 1},
                "endBinding": {"elementId": to_id, "focus": 0, "gap": 1},
                "startArrowhead": None,
                "endArrowhead": "arrow",
                "roundness": {"type": 2},
            }
        )
        if label:
            lid = self._id("al")
            mx = (sx + ex) // 2 + 6
            my = (sy + ey) // 2 - 12
            self.elements.append(
                _text(lid, mx, my, 160, 18, label, font_size=12, align="left", color="#555")
            )
        return aid

    def note(self, x: int, y: int, w: int, h: int, text: str, color: str = "#fafafa") -> str:
        nid = self._id("n")
        rect = {
            **_base(nid, "rectangle", x, y, w, h),
            "backgroundColor": color,
            "fillStyle": "hachure",
            "strokeWidth": 1,
            "strokeStyle": "dashed",
            "roundness": {"type": 3},
        }
        self.elements.append(rect)
        tid = self._id("nt")
        self.elements.append(
            _text(tid, x + 10, y + 10, w - 20, h - 20, text, font_size=11, align="left")
        )
        return nid

    def label(self, text: str, x: int, y: int, w: int = 300, h: int = 20, size: int = 12) -> str:
        lid = self._id("lab")
        self.elements.append(_text(lid, x, y, w, h, text, font_size=size, align="left"))
        return lid

    def build(self, source_label: str = "toy-torch") -> dict:
        return {
            "type": "excalidraw",
            "version": 2,
            "source": source_label,
            "elements": self.elements,
            "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
            "files": {},
        }


def _base(eid: str, etype: str, x: int, y: int, w: int, h: int) -> dict:
    return {
        "id": eid,
        "type": etype,
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "angle": 0,
        "strokeColor": PALETTE["stroke"],
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 1,
        "strokeStyle": "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": None,
        "seed": 100000 + hash(eid) % 900000,
        "versionNonce": 100000 + hash(eid) % 900000,
        "isDeleted": False,
        "boundElements": None,
        "updated": 1,
        "link": None,
        "locked": False,
    }


def _text(
    eid: str,
    x: int,
    y: int,
    w: int,
    h: int,
    text: str,
    font_size: int = 16,
    align: str = "left",
    color: str = PALETTE["text"],
) -> dict:
    base = _base(eid, "text", x, y, w, h)
    base.update(
        {
            "strokeColor": color,
            "text": text,
            "originalText": text,
            "fontSize": font_size,
            "fontFamily": 1,
            "textAlign": align,
            "verticalAlign": "top",
            "containerId": None,
            "baseline": int(font_size * 0.85),
            "lineHeight": 1.25,
        }
    )
    return base


def write(name: str, data: dict) -> None:
    path = os.path.join(OUT_DIR, name + ".excalidraw")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {path}")


# -------------------------------------------------------------------------
# Diagram 1 — v0.1 core architecture
# -------------------------------------------------------------------------
def diagram_v01_core() -> None:
    b = Builder()
    b.title("toy-torch v0.1 — core architecture", 50, 20)
    b.subtitle(
        "Tape (eager) and Graph IR (compile target) share the Op vocabulary",
        50, 75,
    )

    # Top row — layers
    shape = b.box(60, 140, 180, 70, "Shape", "strides + broadcast\nsrc/shape.rs",
                  color=PALETTE["input"])
    tensor = b.box(280, 140, 180, 70, "Tensor", "CPU f32 storage\nsrc/tensor.rs",
                   color=PALETTE["input"])
    op = b.box(500, 140, 180, 70, "Op",
               "Leaf/Const/Add/Sub/Mul/\nNeg/Relu/Sum\nsrc/op.rs",
               color=PALETTE["input"])

    # Middle row — tape
    tape = b.box(60, 270, 400, 90, "Tape",
                 "arena of TensorId, explicit &mut\nsrc/tape.rs",
                 color=PALETTE["transform"])
    autograd = b.box(500, 270, 260, 90, "Autograd",
                     "reverse-mode backward()\nunbroadcast-aware\nsrc/autograd.rs",
                     color=PALETTE["transform"])

    # Bottom row — graph
    tracer = b.box(60, 420, 180, 70, "Tracer", "Tape → Graph\nsrc/graph/tracer.rs",
                   color=PALETTE["compiler"])
    graph = b.box(280, 420, 180, 70, "Graph IR", "Node + NodeId\nsrc/graph/node.rs",
                  color=PALETTE["compiler"])
    const_fold = b.box(500, 420, 130, 70, "constant_fold",
                       "src/graph/compile.rs",
                       color=PALETTE["compiler"])
    dce = b.box(650, 420, 110, 70, "DCE",
                "dead_code_elim",
                color=PALETTE["compiler"])

    eval_box = b.box(280, 550, 180, 60, "Graph::eval", "runs the IR",
                     color=PALETTE["output"])

    # Arrows
    b.arrow(shape, tensor, (240, 175), (280, 175), "")
    b.arrow(tensor, tape, (370, 210), (260, 270))
    b.arrow(op, tape, (590, 210), (370, 270))

    b.arrow(tape, autograd, (460, 315), (500, 315), "loss id")
    b.arrow(tape, tracer, (150, 360), (150, 420), "trace(outputs)")
    b.arrow(tracer, graph, (240, 455), (280, 455))
    b.arrow(graph, const_fold, (460, 455), (500, 455))
    b.arrow(const_fold, dce, (630, 455), (650, 455))
    b.arrow(dce, eval_box, (705, 490), (460, 550))
    b.arrow(const_fold, eval_box, (565, 490), (370, 550))

    b.note(60, 620, 700, 60,
           "Design decision (pedagogy): tape is explicit &mut Tape at every op site — not Rc<RefCell>, not thread_local. "
           "Makes dispatch visible in the type system; graph IR is separate so passes don't accidentally execute.")

    write("01 — v0.1 core architecture", b.build())


# -------------------------------------------------------------------------
# Diagram 2 — v0.2 fusion
# -------------------------------------------------------------------------
def diagram_v02_fusion() -> None:
    b = Builder()
    b.title("toy-torch v0.2 — elementwise fusion", 50, 20)
    b.subtitle(
        "Single-use chains collapse into Op::Fused with a FusedRecipe expression tree",
        50, 75,
    )

    # Before fusion
    b.label("Before (traced 10 nodes)", 60, 130, size=14)
    x1 = b.box(60, 160, 110, 50, "Leaf x", color=PALETTE["input"])
    y1 = b.box(60, 220, 110, 50, "Leaf y", color=PALETTE["input"])
    z1 = b.box(60, 280, 110, 50, "Leaf z", color=PALETTE["input"])
    k1 = b.box(60, 340, 110, 50, "Const 2", color=PALETTE["input"])
    sub1 = b.box(200, 160, 110, 50, "Sub", color=PALETTE["transform"])
    mul1 = b.box(340, 160, 110, 50, "Mul", color=PALETTE["transform"])
    relu1 = b.box(200, 230, 110, 50, "Relu", color=PALETTE["transform"])
    mul2 = b.box(340, 280, 110, 50, "Mul (x·2)", color=PALETTE["transform"])
    add1 = b.box(200, 340, 110, 50, "Add", color=PALETTE["transform"])
    sum1 = b.box(60, 420, 250, 50, "Sum (output)", color=PALETTE["output"])

    b.arrow(x1, sub1, (170, 185), (200, 185))
    b.arrow(y1, sub1, (170, 245), (200, 200))
    b.arrow(sub1, mul1, (310, 185), (340, 185))
    b.arrow(z1, mul1, (170, 305), (370, 210))
    b.arrow(mul1, relu1, (395, 210), (255, 230))
    b.arrow(x1, mul2, (170, 185), (340, 290))
    b.arrow(k1, mul2, (170, 365), (395, 330))
    b.arrow(relu1, add1, (255, 280), (255, 340))
    b.arrow(mul2, add1, (340, 330), (310, 360))
    b.arrow(add1, sum1, (255, 390), (185, 420))

    # Arrow pointing to "after"
    b.label("→ fuse_elementwise →", 480, 280, w=200, size=16)

    # After fusion
    b.label("After fusion (4 live nodes)", 680, 130, size=14)
    x2 = b.box(680, 160, 100, 45, "Leaf x", color=PALETTE["input"])
    y2 = b.box(800, 160, 100, 45, "Leaf y", color=PALETTE["input"])
    z2 = b.box(680, 220, 100, 45, "Leaf z", color=PALETTE["input"])
    k2 = b.box(800, 220, 100, 45, "Const 2", color=PALETTE["input"])

    fused = b.box(680, 295, 220, 110, "Op::Fused",
                  "recipe: Add(\n  Relu(Mul(Sub(x,y),z)),\n  Mul(x,2))",
                  color=PALETTE["highlight"])
    sum2 = b.box(680, 430, 220, 50, "Sum", color=PALETTE["output"])

    b.arrow(x2, fused, (730, 205), (730, 295))
    b.arrow(y2, fused, (850, 205), (820, 295))
    b.arrow(z2, fused, (730, 265), (760, 295))
    b.arrow(k2, fused, (850, 265), (860, 295))
    b.arrow(fused, sum2, (790, 405), (790, 430))

    b.note(60, 510, 840, 80,
           "Rules (src/graph/fusion.rs):\n"
           "• Only Op::is_fuseable() (Add/Sub/Mul/Neg/Relu). Matmul / Sum act as external inputs.\n"
           "• A producer is absorbed into a consumer only if it has exactly ONE consumer. Multi-use nodes stay materialized to avoid duplicating work.\n"
           "• FusedRecipe.expr is an Expr tree (Input(k) / Const / Add / Sub / Mul / Neg / Relu); eval walks the tree per output element.")

    write("02 — v0.2 fusion", b.build())


# -------------------------------------------------------------------------
# Diagram 3 — v0.3 codegen demo (emit_rust)
# -------------------------------------------------------------------------
def diagram_v03_codegen() -> None:
    b = Builder()
    b.title("toy-torch v0.3 — emit_rust codegen (demo)", 50, 20)
    b.subtitle(
        "Walk a FusedRecipe's Expr tree and emit equivalent Rust source",
        50, 75,
    )

    # Input: FusedRecipe tree
    b.label("Input: FusedRecipe.expr", 60, 140, size=14)
    recipe = b.box(60, 170, 260, 220,
                   "Add(\n  Relu(\n    Mul(\n      Sub(Input(0), Input(1)),\n      Input(2))),\n  Mul(Input(0), Input(3)))",
                   color=PALETTE["transform"])

    b.label("emit_rust(recipe, n_inputs=4)", 380, 250, w=280, size=14)
    b.label("src/graph/codegen.rs", 380, 275, w=280, size=11, h=16)
    b.arrow(recipe, recipe, (320, 280), (378, 260))  # hack — arrow label indicator

    # Output: emitted Rust
    b.label("Output: Rust source (string)", 660, 140, size=14)
    emitted = b.box(660, 170, 540, 260,
                    "pub fn kernel(inputs: &[&[f32]], output: &mut [f32]) {\n"
                    "  assert_eq!(inputs.len(), 4);\n"
                    "  let n = output.len();\n"
                    "  let in0 = inputs[0]; let in1 = inputs[1];\n"
                    "  let in2 = inputs[2]; let in3 = inputs[3];\n"
                    "  for i in 0..n {\n"
                    "    output[i] = (\n"
                    "      ((in0[i]-in1[i]) * in2[i]).max(0.0)\n"
                    "      + (in0[i] * in3[i])\n"
                    "    );\n"
                    "  }\n"
                    "}",
                    color=PALETTE["output"])

    # Big arrow recipe -> emitted
    b.arrow(recipe, emitted, (320, 280), (660, 300), "emit_rust")

    b.note(60, 470, 1140, 110,
           "Pedagogical, not wired: v0.3 shows the codegen SHAPE (walk IR → emit target-language text) without actually compiling the output.\n"
           "In v0.5 the same walk is done in emit_c, and the emitted C is compiled with cc -O3 -shared -fPIC and dlopen'd via libloading — closing the interpreter gap for real.\n"
           "Real ML compilers follow the same pattern: TorchInductor emits Triton, XLA emits HLO→LLVM, TVM emits LLVM. None interpret their IR.")

    write("03 — v0.3 codegen demo", b.build())


# -------------------------------------------------------------------------
# Diagram 4 — v0.4 benchmarks (bar chart shape)
# -------------------------------------------------------------------------
def diagram_v04_benchmarks() -> None:
    b = Builder()
    b.title("toy-torch v0.4 → v0.5 — benchmark results", 50, 20)
    b.subtitle(
        "Memory-bound elementwise chain,  out = relu((x-y)*z) + x*k,  N = 1,048,576",
        50, 75,
    )

    # Bar chart — throughput in Gelem/s, max ~6.5
    base_y = 500
    bar_w = 160
    gap = 50
    start_x = 100

    data = [
        ("eager_vec",      2.1, PALETTE["bad"],       "5 loops, 5 Vec<f32>\nintermediates"),
        ("fused_interp",   0.088, PALETTE["bad"],     "Graph::eval() tree walk\n(per-element match)"),
        ("hand_fused",     5.9, PALETTE["good"],      "emit_c shape by hand;\none tight loop"),
        ("fused_jit",      6.5, PALETTE["highlight"], "emit_c + cc -O3\n+ dlopen (v0.5)"),
    ]

    max_val = 7.0
    chart_h = 300

    for i, (name, val, color, desc) in enumerate(data):
        x = start_x + i * (bar_w + gap)
        h = int(val / max_val * chart_h)
        y = base_y - h
        b.box(x, y, bar_w, h, f"{val:g}", desc, color=color)
        b.label(name, x, base_y + 15, w=bar_w, size=14)

    # Y axis label
    b.label("Throughput (Gelem/s)", 100, 180, size=14)

    # Baseline row
    b.label("fused_jit is 74× faster than fused_interp and slightly beats hand_fused",
            100, base_y + 60, w=1000, size=16)
    b.label(
        "cc -O3 auto-vectorizes the emitted C more aggressively than rustc's default on hand-rolled Rust.",
        100, base_y + 85, w=1000, size=12)

    b.note(100, base_y + 120, 1000, 90,
           "Lesson: fusion STRUCTURE alone is a speed loss (interp = 22× slower than eager) because per-element match dispatch dominates.\n"
           "Fusion + codegen is a 74× speed win.  This is exactly the gap real ML compilers close:\n"
           "  TorchInductor → Triton kernels   |   XLA → LLVM IR   |   TVM → LLVM IR.  None interpret their IR.")

    write("04 — v0.4 to v0.5 benchmarks", b.build())


# -------------------------------------------------------------------------
# Diagram 5 — v0.5 JIT pipeline
# -------------------------------------------------------------------------
def diagram_v05_jit() -> None:
    b = Builder()
    b.title("toy-torch v0.5 — JIT pipeline (emit_c + cc + dlopen)", 50, 20)
    b.subtitle(
        "FusedRecipe → temp .c file → cc -O3 → .dylib → libloading → kernel function pointer",
        50, 75,
    )

    # Horizontal pipeline
    y = 220
    h = 90

    n1 = b.box(50,   y, 160, h, "FusedRecipe",
               "post-fusion\nexpression tree",
               color=PALETTE["transform"])
    n2 = b.box(240,  y, 160, h, "emit_c()",
               "walk Expr →\nRust string of C",
               color=PALETTE["compiler"])
    n3 = b.box(430,  y, 160, h, "/tmp/...c",
               "fs::write\n(unique name)",
               color=PALETTE["compiler"])
    n4 = b.box(620,  y, 180, h, "Command::new(\"cc\")",
               "-O3 -shared -fPIC\nproduces .dylib",
               color=PALETTE["compiler"])
    n5 = b.box(830,  y, 160, h, "Library::new",
               "dlopen\n(libloading)",
               color=PALETTE["compiler"])
    n6 = b.box(1020, y, 170, h, "lib.get::<KernelFn>",
               "resolve `kernel`\nsymbol",
               color=PALETTE["compiler"])

    b.arrow(n1, n2, (210, y + 45), (240, y + 45))
    b.arrow(n2, n3, (400, y + 45), (430, y + 45))
    b.arrow(n3, n4, (590, y + 45), (620, y + 45))
    b.arrow(n4, n5, (800, y + 45), (830, y + 45))
    b.arrow(n5, n6, (990, y + 45), (1020, y + 45))

    # JitKernel struct
    b.label("returns ↓", 1080, y + h + 10, size=12)
    jit = b.box(900, y + h + 40, 290, 110, "JitKernel",
                "fn_ptr: KernelFn  // dropped first\n_lib: Library      // dropped last\n_lib_path: PathBuf",
                color=PALETTE["output"])

    # Call path
    call = b.box(100, y + h + 40, 700, 110, "jit.call(inputs, output)",
                 "inputs: &[&[f32]]\n build Vec<*const f32> of pointers\n unsafe { (self.fn_ptr)(n_inputs, ptr_array, out_ptr, n); }",
                 color=PALETTE["highlight"])
    b.arrow(jit, call, (900, y + h + 95), (800, y + h + 95))

    # Safety note
    b.note(50, y + h + 180, 1140, 100,
           "Soundness sketch (src/graph/jit.rs):  the kernel function pointer is only valid while the Library is alive.\n"
           "Struct field order — fn_ptr declared BEFORE _lib — means Drop runs fn_ptr first (no-op on raw pointer) then _lib (unmaps the dylib).\n"
           "Function pointer never escapes JitKernel; `call` synthesizes the ABI at each call site with an unsafe block.")

    b.note(50, y + h + 290, 1140, 60,
           "Mirrors TorchInductor's pattern (emit Triton → compile to .cubin → dlopen → launch), but on CPU with C — same shape, simpler target.")

    write("05 — v0.5 JIT pipeline", b.build())


# -------------------------------------------------------------------------
# Diagram 6 — v0.7 reduction fusion
# -------------------------------------------------------------------------
def diagram_v07_reduction() -> None:
    b = Builder()
    b.title("toy-torch v0.7 — reduction fusion", 50, 20)
    b.subtitle(
        "Sum(Fused(...)) collapses into Op::FusedSum — one pass, scalar accumulator, no intermediate buffer",
        50, 75,
    )

    # Before
    b.label("Before fuse_reductions", 60, 140, size=14)

    xb = b.box(60, 170, 120, 50, "Leaf x", color=PALETTE["input"])
    yb = b.box(200, 170, 120, 50, "Leaf y", color=PALETTE["input"])
    zb = b.box(60, 240, 120, 50, "Leaf z", color=PALETTE["input"])
    kb = b.box(200, 240, 120, 50, "Const 2", color=PALETTE["input"])

    fused_b = b.box(60, 320, 260, 90, "Op::Fused",
                    "recipe writes\nN outputs to buffer",
                    color=PALETTE["transform"])

    sum_b = b.box(60, 440, 260, 50, "Op::Sum",
                  "reduce buffer → scalar",
                  color=PALETTE["output"])

    b.arrow(xb, fused_b, (120, 220), (120, 320))
    b.arrow(yb, fused_b, (260, 220), (260, 320))
    b.arrow(zb, fused_b, (120, 290), (150, 320))
    b.arrow(kb, fused_b, (260, 290), (230, 320))
    b.arrow(fused_b, sum_b, (190, 410), (190, 440))

    # Arrow across
    b.label("→ fuse_reductions →", 420, 320, w=200, size=16)
    b.label("src/graph/reduction.rs", 420, 345, w=200, size=11, h=16)

    # After
    b.label("After (Op::FusedSum)", 680, 140, size=14)

    xa = b.box(680, 170, 120, 50, "Leaf x", color=PALETTE["input"])
    ya = b.box(820, 170, 120, 50, "Leaf y", color=PALETTE["input"])
    za = b.box(680, 240, 120, 50, "Leaf z", color=PALETTE["input"])
    ka = b.box(820, 240, 120, 50, "Const 2", color=PALETTE["input"])

    fs = b.box(680, 320, 260, 170, "Op::FusedSum",
               "recipe + scalar\naccumulator\n\nfor i in 0..n {\n  acc += eval_at(expr, i);\n}\noutput = scalar(acc)",
               color=PALETTE["highlight"])

    b.arrow(xa, fs, (740, 220), (740, 320))
    b.arrow(ya, fs, (880, 220), (880, 320))
    b.arrow(za, fs, (740, 290), (770, 320))
    b.arrow(ka, fs, (880, 290), (850, 320))

    b.note(60, 560, 880, 110,
           "Gating rules (src/graph/reduction.rs):\n"
           "• Sum's single input must be Op::Fused\n"
           "• Fused node must have exactly one consumer (the Sum) — otherwise absorbing it would duplicate work elsewhere in the graph\n"
           "• FusedRecipe::eval_sum() implements the scalar-accumulator path; no per-element output buffer is ever allocated\n"
           "• Pattern lifted from TorchInductor's reduction-fusion pass — same idea, simpler scope (v0.7 only handles Sum)")

    write("06 — v0.7 reduction fusion", b.build())


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    diagram_v01_core()
    diagram_v02_fusion()
    diagram_v03_codegen()
    diagram_v04_benchmarks()
    diagram_v05_jit()
    diagram_v07_reduction()
    print("all diagrams written")


if __name__ == "__main__":
    main()
