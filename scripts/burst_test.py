import os, sys, time
from concurrent.futures import ThreadPoolExecutor
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(here, ".env"))
from openai import OpenAI

def burst(name, key, base, model, n):
    c = OpenAI(api_key=key, base_url=base)
    def one(i):
        try:
            r = c.chat.completions.create(model=model, max_tokens=10,
                messages=[{"role": "user", "content": "Say hi."}])
            return ("ok", (r.choices[0].message.content or "")[:10])
        except Exception as e:
            return ("ERR", str(e)[:80])
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n) as ex:
        res = list(ex.map(one, range(n)))
    ok = sum(1 for s, _ in res if s == "ok")
    errs = [m for s, m in res if s != "ok"][:3]
    print(f"{name} x{n}: {ok}/{n} ok in {time.time()-t0:.1f}s {errs}")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    burst("opusgate-sol", os.environ["OPUSKEY"], "https://api.opusgate.dev/v1", "gpt-5.6-sol", n)
    burst("minirouter-luna", os.environ["MINIROUTER_KEY"], "https://api.minirouter.sh/v1", "openai/gpt-5.6-luna", n)
