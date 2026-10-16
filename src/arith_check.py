"""Extract arithmetic equalities from a math chain of thought and check them exactly with Fractions.

How it works. LaTeX operators are normalised to Python ones. A *run* is a maximal stretch of
digits, operators, parentheses, decimal points and '=' signs. Each run is split on '=' into
segments; every adjacent pair where both segments evaluate and at least one contains an
operator is one checked equation. So the chain  7^4 + 5*7^3 = 2401 + 1715 = 4116  checks the
two steps (7^4 + 5*7^3, 2401 + 1715) and (2401 + 1715, 4116), and  2x + 3 = 7  is never checked:
the x breaks the run, and a segment glued to a letter, brace or backslash is discarded.

Everything is linear in the text length (character-class regexes, an ast walk per segment), so
a pathological sample cannot stall grading. Exponents are capped so 10^(10^10) is skipped.

check(text) -> (n_checked, n_wrong, details)  with details = [(lhs, rhs, lhs_value, rhs_value, ok), ...]
"""
import ast
import re
from fractions import Fraction

# Normalise common LaTeX / unicode operators to Python before scanning. Math delimiters and layout
# scaffolding become separators first: before 2026-10-15 \\[ \\] \\( \\) counted as glue, so the first
# and last side of nearly every displayed equation was discarded (coverage 22-26% of AIME samples).
_NORM = [
    (r'\\boxed\{([^{}]*)\}', r' \1 '),
    (r'\\\[|\\\]|\\\(|\\\)|\$\$', ' ; '),
    (r'\\(?:quad|qquad|implies|Rightarrow|Longrightarrow|iff|therefore|hline|nonumber)\b'
     r'|\\text\{[^{}]*\}|\\begin\{[a-z*]+\}|\\end\{[a-z*]+\}|\\\\', ' ; '),
    (r'&', ''),                                     # align column marker: sits next to the '=', just drop it
    (r'\\times|\\cdot|×|·|\*\*?', '*'),
    (r'\\div|÷', '/'),
    (r'\\left\(|\\right\)', lambda m: '(' if 'left' in m.group(0) else ')'),
    (r'\\d?frac\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))'),
    (r'\^\{([^{}]+)\}', r'^(\1)'),
    (r'\\,|\\!|\\;|\\:|\\ ', ''),   # LaTeX spacing; was r'\;' which deleted every literal semicolon
    (r'\\%|%', ''),
    (r'\$', ' '),
    (r'−|–', '-'),
    (r'\\approx|≈', '='),
    (r'(?<=\d),(?=\d{3}\b)', ''),      # thousands separators: 1,234 -> 1234
    (r'\.(?!\d)', ' ; '),             # a period not followed by a digit ends a sentence, not a number
    (r',', ' ; '),                    # any remaining comma is a list separator
]
_RUN = re.compile(r'[\d\s+\-*/^().=]+')
_SEG_START = re.compile(r'^[-(]?\s*[\d(]')
_SEG_END = re.compile(r'[\d)]$')
_HAS_OP = re.compile(r'[\d)]\s*[-+*/^]')
_GLUE = set('\\{}[]_')                 # a run touching one of these (or a letter/digit) is contaminated
MAX_EXP = 1000
MAX_BITS = 8192


def normalise(text):
    for pat, rep in _NORM:
        text = re.sub(pat, rep, text)
    return text


def _ev(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return Fraction(str(node.value)) if isinstance(node.value, float) else Fraction(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _ev(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp):
        a, b = _ev(node.left), _ev(node.right)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Div):
            return a / b                      # ZeroDivisionError propagates: the pair is skipped
        if isinstance(node.op, ast.Pow):
            if b.denominator != 1 or abs(b) > MAX_EXP:
                raise ValueError('exponent')
            if max(a.numerator.bit_length(), a.denominator.bit_length()) * abs(int(b)) > MAX_BITS:
                raise ValueError('too big')
            return a ** int(b)
    raise ValueError('unsupported')


def _value(seg):
    seg = seg.strip()
    if not seg or len(seg) > 400 or not _SEG_START.match(seg) or not _SEG_END.search(seg):
        raise ValueError('not an expression')
    return _ev(ast.parse(seg.replace('^', '**'), mode='eval').body)


def _ok(v, r, lhs, rhs, rel_tol):
    if v == r:
        return True
    try:
        dec = max((len(t.split('.')[1]) for t in re.findall(r'\d+\.\d+', lhs + ' ' + rhs)), default=0)
        if dec and abs(float(v) - float(r)) <= 0.5 * 10 ** -dec + 1e-12:
            return True
        return r != 0 and abs(float(v - r) / float(r)) <= rel_tol
    except OverflowError:
        return False


def _glued_before(prefix):
    """Is the text right before a run a math token (variable, subscript, \\command, brace)? Prose words
    such as 'So' or 'and' are not; a single letter, anything with a backslash, or a brace/bracket is."""
    if not prefix:
        return False
    if prefix[-1].isalnum() or prefix[-1] in _GLUE:
        return True
    w = re.search(r'(\S+)\s*$', prefix)
    w = w.group(1) if w else ''
    return bool(w) and ((len(w) == 1 and w.isalpha()) or w[-1] in _GLUE or '\\' in w)


def _glued_after(suffix):
    if not suffix:
        return False
    if suffix[0].isalnum() or suffix[0] in _GLUE:
        return True
    w = re.match(r'\s*(\S+)', suffix)
    w = w.group(1) if w else ''
    return bool(w) and ((len(w) == 1 and w.isalpha()) or w[0] in _GLUE or w.startswith('\\'))


def check(text, rel_tol=1e-6):
    """Identical equations within one text are counted once, so a looping sample that repeats a
    false line 100 times contributes one error, not 100."""
    text = normalise(text)
    n = wrong = 0
    details = []
    seen = set()
    for m in _RUN.finditer(text):
        run = m.group(0)
        if '=' not in run:
            continue
        start = m.start() + (len(run) - len(run.lstrip()))
        end = m.end() - (len(run) - len(run.rstrip()))
        segs = run.split('=')
        if _glued_before(text[:start]):
            segs[0] = ''
        if _glued_after(text[end:]):
            segs[-1] = ''
        vals = []
        for seg in segs:
            try:
                vals.append((seg.strip(), _value(seg)))
            except Exception:  # noqa: BLE001  (syntax, unsupported node, zero division, huge power)
                vals.append(None)
        for (l, r) in zip(vals, vals[1:]):
            if l is None or r is None or not (_HAS_OP.search(l[0]) or _HAS_OP.search(r[0])):
                continue
            key = (re.sub(r'\s+', '', l[0]), re.sub(r'\s+', '', r[0]))
            if key in seen:
                continue
            seen.add(key)
            ok = _ok(l[1], r[1], l[0], r[0], rel_tol)
            n += 1
            wrong += (not ok)
            details.append((l[0], r[0], str(l[1]), str(r[1]), ok))
    return n, wrong, details


if __name__ == '__main__':
    for s in [
        "So 12 * 13 = 156 and 156 + 7 = 163.",
        "Wrong: 17 * 23 = 381. Also 100 - 37 = 73 is wrong too.",
        "$7^4 + 5 \\cdot 7^3 + 9 \\cdot 7^2 - 7 - 14 = 2401 + 1715 + 441 - 7 - 14 = 4536$",
        "Equation x + 3 = 5 should not match; 2x = 4 should not match; x - 3 = 7 neither.",
        "Rounded: 22/7 = 3.14 is fine; 1/3 = 0.333. Thus 4 + 4 = 8 = 2 * 4.",
    ]:
        n, w, d = check(s)
        print(f"n={n} wrong={w} :: {s}")
        for x in d:
            print("    ", x)
