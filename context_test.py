#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Context-length stress test for a local MLX (OpenAI-compatible) server.

Runs a "needle-in-a-haystack" sweep: hides a secret code near the start of the
prompt, pads with filler up to a target size, then asks the model to recall the
code. Reports, for each size:

  * the ACTUAL prompt token count the server saw (from response usage),
  * whether the request succeeded or the server errored (hard limit / OOM),
  * whether the model correctly recalled the needle (effective context),
  * how long the request took.

This finds BOTH limits:
  - HARD limit  -> first size that returns a server error.
  - EFFECTIVE   -> first size where recall flips from OK to FAIL.

Usage:
    uv run context_test.py
    uv run context_test.py --model mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit
    uv run context_test.py --port 8080 --sizes 1000,4000,16000,64000,131072
"""
import argparse
import json
import random
import string
import sys
import time
import urllib.error
import urllib.request

DEFAULT_SIZES = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000, 131072]

# A pool of varied filler sentences so the haystack isn't trivially compressible.
FILLER_SENTENCES = [
    "The logistics report indicated steady throughput across all regional hubs.",
    "Quarterly metrics were reviewed and archived for future reference.",
    "A gentle rain fell over the harbor as the ferries changed their schedule.",
    "Engineers noted the calibration drift and scheduled a maintenance window.",
    "The committee deferred the vote pending additional supplier quotations.",
    "Migratory patterns shifted earlier this year according to field observers.",
    "Inventory counts reconciled cleanly after the overnight batch process ran.",
    "The lecture covered thermodynamics before moving on to fluid mechanics.",
    "Travelers were advised to allow extra time during the holiday period.",
    "The garden's irrigation system was upgraded to a low-flow drip design.",
]


def make_prompt(approx_tokens: int, needle_code: str, depth: float) -> str:
    """Build a prompt of ~approx_tokens tokens with the needle inserted at `depth`
    (0.0 = very start, 0.5 = middle, 1.0 = just before the trailing question)."""
    needle = (
        f" IMPORTANT FACT: The secret passcode is {needle_code}. "
        f"You must remember this exact passcode. "
    )
    tail = (
        "\n\nQUESTION: What is the secret passcode stated somewhere in this "
        "document? Reply with ONLY the passcode, nothing else."
    )

    # ~0.75 words per token is a reasonable English approximation; the script
    # reports the *real* token count from the server afterwards regardless.
    target_words = int(approx_tokens * 0.75)
    words = []
    i = 0
    while len(words) < target_words:
        sentence = f"[para {i:05d}] " + random.choice(FILLER_SENTENCES)
        words.extend(sentence.split())
        i += 1
    words = words[:target_words]

    insert_at = max(0, min(int(len(words) * depth), len(words)))
    body = " ".join(words[:insert_at]) + needle + " ".join(words[insert_at:])
    return body + tail


def call_server(base_url: str, model: str, prompt: str, timeout: int):
    """Returns (ok, prompt_tokens, content, error_str, elapsed)."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.0,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        method="POST",
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        elapsed = time.time() - start
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {}) or {}
        # "thinking" models may put the answer in reasoning_content, or use text.
        content = (msg.get("content")
                   or msg.get("reasoning_content")
                   or choice.get("text")
                   or "")
        ptoks = data.get("usage", {}).get("prompt_tokens")
        return True, ptoks, content, None, elapsed
    except urllib.error.HTTPError as e:
        elapsed = time.time() - start
        detail = e.read().decode("utf-8", "ignore")[:200]
        return False, None, None, f"HTTP {e.code}: {detail}", elapsed
    except Exception as e:  # timeout, connection reset (often OOM/crash), etc.
        elapsed = time.time() - start
        return False, None, None, f"{type(e).__name__}: {e}", elapsed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--model", default=None,
                    help="Model ID. Defaults to whatever the server reports first.")
    ap.add_argument("--sizes", default=None,
                    help="Comma-separated approx token targets (e.g. 1000,8000,64000).")
    ap.add_argument("--timeout", type=int, default=600,
                    help="Per-request timeout in seconds (default 600).")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}/v1"

    # Discover the model if not given.
    model = args.model
    if not model:
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=10) as r:
                models = json.loads(r.read().decode())["data"]
            model = models[0]["id"]
            print(f"  Using model (first reported): {model}")
            if len(models) > 1:
                print(f"  Tip: server also has {len(models) - 1} other model(s); "
                      f"pass --model to pick one.\n")
        except Exception as e:
            print(f"  Could not reach {base_url}/models : {e}")
            print("  Is the server running?  uv run python start.py")
            sys.exit(1)

    sizes = ([int(s) for s in args.sizes.split(",")] if args.sizes else DEFAULT_SIZES)

    print(f"\n  Target server : {base_url}")
    print(f"  Model         : {model}")
    print(f"  Sizes (approx): {sizes}")
    print(f"  Needle depths : start, 25%, 50%, 75%, end\n")

    depths = [0.0, 0.25, 0.50, 0.75, 1.0]
    labels = ["start", "25%", "50%", "75%", "end"]

    print(f"  {'real_tok':>9}   " + "  ".join(f"{l:>5}" for l in labels) + "    secs")
    print("  " + "-" * 54)

    # each entry: (size, real_tok, marks_dict, all_ok, any_fail)
    results = []
    hard_limit_tokens = None
    last_full_pass_tokens = None

    for size in sizes:
        marks = {}
        real_tok = None
        total_secs = 0.0
        had_error = False
        err_detail = None
        for depth, label in zip(depths, labels):
            code = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
            prompt = make_prompt(size, code, depth)
            ok, ptoks, content, err, elapsed = call_server(base_url, model, prompt, args.timeout)
            total_secs += elapsed
            if not ok:
                marks[label] = "ERR"
                had_error = True
                err_detail = err
                continue
            if ptoks:
                real_tok = ptoks
            marks[label] = "\u2713" if code in (content or "") else "\u2717"

        tok_str = f"{real_tok:,}" if real_tok else f"~{size:,}"
        row = "  ".join(f"{marks.get(l, '?'):>5}" for l in labels)
        print(f"  {tok_str:>9}   {row}    {total_secs:>5.1f}")

        all_ok = all(marks.get(l) == "\u2713" for l in labels)
        any_fail = any(marks.get(l) == "\u2717" for l in labels)
        results.append((size, real_tok, marks, all_ok, any_fail))
        if all_ok and real_tok:
            last_full_pass_tokens = real_tok

        if had_error:
            hard_limit_tokens = last_full_pass_tokens
            print(f"       \u21b3 {err_detail}")
            print("\n  Server stopped accepting this size \u2014 treating as the hard limit.")
            break

    # ---- Final report ---------------------------------------------------------
    print("\n  " + "=" * 60)
    print("  REPORT")
    print("  " + "=" * 60)

    full_pass = [r for r in results if r[3] and r[1]]
    first_degrade = next((r for r in results if r[4]), None)

    if full_pass:
        print(f"  \u2022 Reliable at ALL depths up to : {full_pass[-1][1]:,} tokens")
    else:
        print("  \u2022 Reliable at ALL depths up to : failed even at the smallest size")

    if first_degrade:
        bad = [l for l in labels if first_degrade[2].get(l) == "\u2717"]
        size_tok = first_degrade[1] or first_degrade[0]
        print(f"  \u2022 First degradation           : ~{size_tok:,} tokens "
              f"(missed: {', '.join(bad)})")
    else:
        print("  \u2022 First degradation           : none seen in tested range")

    if hard_limit_tokens is not None:
        print(f"  \u2022 Hard server/RAM limit       : above ~{hard_limit_tokens:,} tokens")
    elif results and any(m == "ERR" for m in results[-1][2].values()):
        print("  \u2022 Hard server/RAM limit       : hit (see error above)")
    else:
        print("  \u2022 Hard server/RAM limit       : not reached in tested range")

    print("\n  RECOMMENDED CONTEXT SIZE")
    print("  " + "-" * 60)
    if full_pass:
        safe = full_pass[-1][1]
        working = int(safe * 0.8)            # leave headroom for the model's reply
        nice = 1
        while nice * 2 <= working:
            nice *= 2
        print(f"  Keep working prompts under ~{working:,} tokens "
              f"(\u2248 {nice:,} is a clean value to target).")
        print("  Always leave room for the response \u2014 don't fill the whole window.")
        print("  Launch the server capped to a safe KV size, e.g.:")
        print(f"      python -m mlx_lm server --model {model} \\")
        print(f"          --max-kv-size {max(nice, 4096)}")
    else:
        print("  Recall failed even at the smallest size \u2014 try a larger/stronger model,")
        print("  or check the server (it may be truncating or mis-loading the model).")
    print()


if __name__ == "__main__":
    main()
