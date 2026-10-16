"""arith_check: chains, non-arithmetic equations, rounding, and that nothing stalls. Run: python tests/test_arith.py"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import arith_check as ac  # noqa: E402


def nw(text):
    n, w, _ = ac.check(text)
    return n, w


def test_simple_and_wrong():
    assert nw("So 12 * 13 = 156 and 156 + 7 = 163.") == (2, 0)
    assert nw("Wrong: 17 * 23 = 381. Also 100 - 37 = 73 is wrong too.") == (2, 2)
    assert nw("We have 12 * 13 = 156 and 17 * 23 = 381.") == (2, 1)
    assert nw("Compute 1,234 + 5,678 = 6,912. Then 6912 / 8 = 864.") == (2, 0)
    assert nw("Hence $2^{10} = 1024$ and $2^5 = 32$ and $\\frac{3}{4} \\cdot 8 = 6$.") == (3, 0)
    assert nw("Percent: 15 * 0.2 = 3, and 0.1 + 0.2 = 0.4 is wrong.") == (2, 1)


def test_chains_are_checked_stepwise():
    assert nw("$7^4 + 5 \\cdot 7^3 + 9 \\cdot 7^2 - 7 - 14 = 2401 + 1715 + 441 - 7 - 14 = 4536$") == (2, 0)
    assert nw("Thus 4 + 4 = 8 = 2 * 4.") == (2, 0)
    assert nw("2 + 3 = 5 + 1 = 6") == (2, 1)
    assert nw("(-14)^4 + 5 * (-14)^3 + 9 * (-14)^2 - (-14) - 14 = 38416 - 13720 + 1764 + 14 - 14 = 26460") == (2, 0)


def test_symbols_do_not_count():
    assert nw("Equation x + 3 = 5 should not match; 2x = 4 should not match; x - 3 = 7 neither.") == (0, 0)
    assert nw("Let a_1 = 3 and n = 5, k = 2. Then a = 2 + 3 = 5.") == (1, 0)
    assert nw("So \\sqrt{2} \\cdot 3 = 4.24 and \\sqrt{16} = 4 and 3 = 3x.") == (0, 0)
    assert nw("We get 5. 3 + 4 = 7.") == (1, 0)


def test_rounding_tolerance():
    assert nw("Rounded: 22/7 = 3.14 is fine; 1/3 = 0.333.") == (2, 0)
    assert nw("22/7 \\approx 3.14") == (0, 0)          # approximations are not exact claims: not checked
    assert nw(r"\(1\cdot 2\cdots 9 = 362880\)") == (0, 0)  # an ellipsis is not one product
    assert nw(r"a_n - 2019\cdot1010 = 2019") == (0, 0)       # the expression started before the run
    assert nw("1/3 = 0.3") == (1, 0)
    assert nw("1/3 = 0.4") == (1, 1)


def test_pathological_inputs_finish_fast():
    t = time.time()
    frac = ("\\binom{8}{2} = \\frac{8 \\cdot 7 \\cdot 6 \\cdot 5 \\cdot 4 \\cdot 3 \\cdot 2 \\cdot 1}"
            "{2 \\cdot 1 \\cdot 6 \\cdot 5 \\cdot 4 \\cdot 3 \\cdot 2 \\cdot 1} = 28")
    assert nw(frac) == (1, 0)                                   # the Colab staller
    assert nw(" * ".join(["1"] * 80) + " = 1") == (1, 0)
    assert nw(" * ".join(["1"] * 3000) + " = 1") == (0, 0)       # over the 400-char segment cap: skipped, fast
    assert nw(" + ".join(["1"] * 3000)) == (0, 0)
    assert nw("((((((((((1 + 2)))))))))) = 3") == (1, 0)
    assert nw("10^{10^{10}} = 1 and 2^{2^{64}} = 4 and 10^100 = 1") == (1, 1)   # huge powers skipped, 10^100 checked
    assert nw("1/0 = 5 and 00123 + 1 = 124") == (0, 0)           # zero division and bad literals skipped
    assert nw("1" * 6000 + " = 2") == (0, 0)                    # no operator: nothing to check
    assert time.time() - t < 2.0


def test_details_format():
    n, w, d = ac.check("17 * 23 = 381")
    assert (n, w) == (1, 1) and d == [("17 * 23", "381", "391", "381", False)]


def test_display_math_boxed_align_semicolons_and_dedupe():
    assert nw(r"\[ \frac{12\cdot13}{2} = 78 \]") == (1, 0)          # delimiters used to glue both ends
    assert nw(r"\(m+n = 85+24 = 109\)") == (1, 0)
    assert nw(r"so \(3+4 = \boxed{7}\)") == (1, 0)
    assert nw(r"\begin{align*} 2+2 &= 4 \\ 3\cdot 3 &= 10 \end{align*}") == (2, 1)
    assert nw("a; 3+4 = 7; b") == (1, 0)                              # semicolons are no longer deleted
    assert nw("1+1 = 3. " * 50) == (1, 1)                              # a looping false line counts once
    assert nw(r"\[ x = 3 \]") == (0, 0)

if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
