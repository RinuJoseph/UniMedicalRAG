
import concurrent.futures as cf
import json
import os
import threading
import time

PRICES = {"gpt-4o": (2.5e-6, 10e-6), "gpt-4o-mini": (1.5e-7, 6e-7)}


def estimate(n_calls, mean_prompt_chars, out_tokens, model):
    cin, cout = PRICES.get(model, PRICES["gpt-4o"])
    tin = mean_prompt_chars / 4
    return n_calls * (tin * cin + out_tokens * cout), tin


def load_cache(path):
    cache = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (r.get("response") or "").strip():
                cache[str(r["id"])] = r["response"]
    return cache


def _one(client, model, system, user, max_tokens):
    for attempt in range(6):
        try:
            msgs = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": user}]
            r = client.chat.completions.create(model=model, temperature=0,
                                               max_tokens=max_tokens, messages=msgs)
            t = r.choices[0].message.content
            if t and t.strip():
                return t, r.usage.prompt_tokens, r.usage.completion_tokens
        except Exception:
            pass
        time.sleep(min(2 ** attempt, 30))
    return "", 0, 0


def run_batch(items, build, cache_path, model, api_key, max_tokens=2000, workers=6):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    cache = load_cache(cache_path)
    todo = [(i, p) for i, p in items if str(i) not in cache]
    print(f"   {len(cache)} cached, {len(todo)} to call, {workers} workers")
    lock, t0, tin, tout = threading.Lock(), time.time(), 0, 0
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as fh:
        def work(item):
            i, p = item
            s, u = build(p)
            return (i,) + _one(client, model, s, u, max_tokens)
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            for n, f in enumerate(cf.as_completed([ex.submit(work, it) for it in todo])):
                i, txt, a, b = f.result()
                with lock:
                    tin += a
                    tout += b
                    cache[str(i)] = txt
                    fh.write(json.dumps({"id": i, "response": txt}, ensure_ascii=False) + "\n")
                    fh.flush()
                    if (n + 1) % 100 == 0:
                        print(f"   {n + 1}/{len(todo)}  ({time.time() - t0:.0f}s)", flush=True)
    cin, cout = PRICES.get(model, PRICES["gpt-4o"])
    print(f"   tokens {tin} in / {tout} out -> ${tin * cin + tout * cout:.2f}")
    return cache
