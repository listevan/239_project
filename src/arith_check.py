"""Prototype: extract 'A op B = C' style arithmetic equalities from a math CoT and
check them exactly with Python Fractions. Returns (n_checked, n_wrong, details)."""
import re
from fractions import Fraction

NUM = r'-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?'
# Normalise common LaTeX/unicode operators to Python before matching.
_NORM = [
    (r'\\times|\\cdot|×|·|\*\*?', '*'),
    (r'\\div|÷', '/'),
    (r'\\left\(|\\right\)', lambda m: '(' if 'left' in m.group(0) else ')'),
    (r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))'),
    (r'\\dfrac\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))'),
    (r'\^\{([^{}]+)\}', r'^(\1)'),
    (r'\\,|\\!|\;|\\ ', ''),
    (r'\\%|%', ''),
    (r'\$', ''),
    (r'−|–', '-'),
]
def normalise(text):
    for pat, rep in _NORM:
        text = re.sub(pat, rep, text)
    return text

# LHS: a chain of numbers/parens/operators containing at least one operator; RHS: a single number.
LHS = r'(?<![\w.])(\(*\s*(?:%s)\s*\)*(?:\s*[-+*/^]\s*\(*\s*(?:%s)\s*\)*)+)' % (NUM, NUM)
# RHS may be followed by sentence punctuation, but not by more digits / a thousands group / a decimal tail
EQ = re.compile(LHS + r'\s*(?:=|\\approx|≈)\s*(%s)(?!\d|,\d{3}|\.\d)' % NUM)

def _to_frac(tok):
    return Fraction(tok.replace(',', ''))

def _eval(expr):
    # tokenise numbers, replace by Fraction literals, evaluate with ^ as power
    expr = expr.replace('^', '**')
    def sub(m): return 'Fraction("%s")' % m.group(0).replace(',', '')
    py = re.sub(NUM, sub, expr)
    if not re.fullmatch(r'[\sFraction("0-9.\-")+*/()\*]*', py):
        raise ValueError('unsafe')
    return eval(py, {'Fraction': Fraction, '__builtins__': {}})

def check(text, rel_tol=1e-6):
    text = normalise(text)
    n = wrong = 0
    details = []
    for m in EQ.finditer(text):
        lhs, rhs = m.group(1), m.group(2)
        try:
            v = _eval(lhs); r = _to_frac(rhs)
        except Exception:
            continue
        # tolerate rounding when RHS has decimals
        decimals = len(rhs.split('.')[1]) if '.' in rhs else 0
        ok = v == r or (decimals > 0 and abs(float(v) - float(r)) <= 0.5 * 10 ** -decimals + 1e-12) \
             or (r != 0 and abs(float(v - r) / float(r)) <= rel_tol)
        n += 1; wrong += (not ok)
        details.append((lhs.strip(), rhs, str(v), ok))
    return n, wrong, details

if __name__ == '__main__':
    samples = [
        "So 12 * 13 = 156 and 156 + 7 = 163.",
        "We get $3 \\times 17 = 51$, then $51 - 9 = 41$, so 41/2 = 20.5.",
        "Compute 1,234 + 5,678 = 6,912. Then 6912 / 8 = 864.",
        "Hence 2^{10} = 1024 and 2^5 = 32.",
        "So \\frac{3}{4} \\cdot 8 = 6.",
        "Wrong: 17 * 23 = 381. Also 100 - 37 = 73 is wrong too.",
        "Rounded: 22/7 = 3.14 is fine; 1/3 = 0.333.",
        "Equation x + 3 = 5 should not match; 2x = 4 should not match.",
        "Percent: 25% of 80 = 20 (not matched, fine). 15 * 0.2 = 3.",
        "Thus 4 + 4 = 8 = 2 * 4.",
    ]
    for s in samples:
        n, w, d = check(s)
        print(f"n={n} wrong={w} :: {s}")
        for x in d: print("    ", x)
