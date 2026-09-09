import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
from openai import OpenAI
from gen_dual import GEN_TPL
which = sys.argv[1] if len(sys.argv) > 1 else "gemini"
if which == "gemini":
    c = OpenAI(api_key=os.environ["MINIROUTER_KEY"], base_url="https://api.minirouter.sh/v1")
    model = "google/gemini-3.8-flash"
else:
    c = OpenAI(api_key=os.environ["OPUSKEY"], base_url="https://api.opusgate.dev/v1")
    model = "gpt-5.6-sol"
prompt = GEN_TPL.replace("{topic}", "phone bills & refunds")
r = c.chat.completions.create(model=model, max_tokens=1500, temperature=0.9,
                              messages=[{"role": "user", "content": prompt}])
content = r.choices[0].message.content or ""
print("LEN", len(content))
print("HEAD:", repr(content[:600]))
print("TAIL:", repr(content[-300:]))
print("USAGE:", r.usage)
