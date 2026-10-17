"""Repair arms whose text was decoded with a broken tokenizer that left byte-level BPE markers ("Ġ" for a
space, "Ċ" for a newline) in the output, as the NuminaMath-7B runs of 2026-10-16 did. The markers map back to
the exact bytes, so this is lossless when every character is a byte-level marker.

Only safe for single-round arms with no tool call and no stop string in the repaired text: there the tokens
the model sampled did not depend on the decoded text, so the repaired text is what a correct run would have
produced. The script refuses anything else. Originals are copied to ~/Downloads/trash first; grades are
cleared so src/grade.py regrades.

Run: uv run tools/repair_bytelevel.py <arm.jsonl> [...]"""
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gen  # noqa: E402


def byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


DEC = byte_decoder()
STOPS = list(gen.TEXT_STOPS) + list(gen.TOOL_STOPS)


def repair_text(text):
    bad = {c for c in text if c not in DEC}
    if bad:
        raise ValueError(f"not byte-level marker text (unmappable {sorted(bad)[:5]})")
    return bytes(DEC[c] for c in text).decode("utf-8")


def repair(path: Path, trash: Path):
    rows = [json.loads(line) for line in open(path)]
    out = []
    for r in rows:
        if r["rounds"] != 1 or r["n_calls"]:
            raise SystemExit(f"{path.name}: {r['problem_id']}#{r['sample_idx']} has {r['rounds']} rounds / "
                             f"{r['n_calls']} calls; its sampling depended on the garbled text, regenerate it")
        text = repair_text(r["text"])
        if any(s in text for s in STOPS):
            raise SystemExit(f"{path.name}: {r['problem_id']}#{r['sample_idx']} contains a stop string the fixed "
                             "harness would have acted on; regenerate it")
        boxed = gen.last_boxed(text)
        out.append({**r, "text": text, "boxed": boxed, "boxed_present": boxed is not None,
                    "code_emitted": "```python" in text or "```py" in text,
                    "fake_outputs": max(0, text.count(gen.STOP_OUTPUT)),
                    "correct": None, "correct_boxed": None, "correct_any": None, "arith_n": None, "arith_wrong": None})
    meta_p = gen.meta_path(path)
    meta = json.loads(meta_p.read_text())
    trash.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, trash / path.name)
    shutil.copy2(meta_p, trash / meta_p.name)
    gen.write_jsonl_atomic(path, out)
    meta.update(graded=False, pass1=None, text_repaired=(
        f"{time.strftime('%Y-%m-%d')}: byte-level markers decoded back to text by tools/repair_bytelevel.py "
        "(the run's tokenizer decoded wrongly; tokens were unaffected: one round, no tool call, no stop string)"))
    meta_p.write_text(json.dumps(meta, indent=1))
    print(f"{path.name}: repaired {len(out)} rows; originals in {trash}")


if __name__ == "__main__":
    trash = Path.home() / "Downloads" / "trash" / f"bytelevel_originals_{time.strftime('%Y%m%d_%H%M')}"
    for p in sys.argv[1:]:
        repair(Path(p), trash)
