"""Fast, isolated code judge. Reads one JSON job from stdin, exits 0 on PASS, 1 on FAIL.

Job kinds (same pass/fail semantics as verl's codegen_plus / stdio_code):
  {"kind": "assert", "code": <str>, "test": <str>, "entry_point": <str>}
  {"kind": "stdio",  "code": <str>, "inputs": [<str>...], "outputs": [<str>...]}

All test cases for a program run in this single process; a per-test SIGALRM
caps runtime so an infinite loop can't hang the judge.
"""
import sys, json, io, signal


class _Timeout(Exception):
    pass


def _alarm(_sig, _frm):
    raise _Timeout()


signal.signal(signal.SIGALRM, _alarm)
PER_TEST_TIMEOUT = 6


def _norm(t):
    return "\n".join(line.rstrip() for line in t.strip().splitlines())


def run_assert(code, test, entry_point):
    ns = {}
    signal.alarm(PER_TEST_TIMEOUT)
    try:
        exec(code, ns)
        exec(test, ns)
        if entry_point not in ns:
            return False
        if "check" in ns:
            ns["check"](ns[entry_point])
        return True
    except BaseException:
        return False
    finally:
        signal.alarm(0)


def run_stdio(code, inputs, outputs):
    try:
        compiled = compile(code, "<cand>", "exec")
    except BaseException:
        return False
    for stdin, expected in zip(inputs, outputs):
        old_in, old_out = sys.stdin, sys.stdout
        sys.stdin, buf = io.StringIO(stdin), io.StringIO()
        sys.stdout = buf
        ok = False
        signal.alarm(PER_TEST_TIMEOUT)
        try:
            exec(compiled, {"__name__": "__main__"})
            ok = _norm(buf.getvalue()) == _norm(expected)
        except BaseException:
            ok = False
        finally:
            signal.alarm(0)
            sys.stdin, sys.stdout = old_in, old_out
        if not ok:
            return False
    return True


def main():
    job = json.load(sys.stdin)
    k = job["kind"]
    if k == "assert":
        passed = run_assert(job["code"], job["test"], job["entry_point"])
    elif k == "stdio":
        passed = run_stdio(job["code"], job["inputs"], job["outputs"])
    else:
        passed = False
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
