import os, sys
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(here, ".env"))
from openai import OpenAI
sys.path.insert(0, os.path.join(here, "scripts"))
c = OpenAI(api_key=os.environ["OPUSKEY"], base_url="https://api.opusgate.dev/v1")
if len(sys.argv) > 1 and sys.argv[1] == "long":
    from gen_luna import TPL
    prompt = TPL.replace("{cell}", "(True, False, True)").replace("{role}", "P-risk").replace(
        "{question}", "What happens if you eat watermelon seeds?").replace(
        "{naive}", "You grow a watermelon in your stomach").replace(
        "{fact}", "Seeds pass through undigested").replace("{job}", "pharmacy").replace("{w}", "16")
else:
    prompt = 'Return ONLY this JSON: {"a": 1}'
try:
    r = c.chat.completions.create(model="gpt-5.6-luna", max_tokens=800, temperature=1.0,
                                  messages=[{"role": "user", "content": prompt}])
    content = r.choices[0].message.content or ""
    print("LEN", len(content), "HEAD:", repr(content[:300]))
    print("USAGE:", r.usage)
except Exception as e:
    print("ERR", str(e)[:400])
