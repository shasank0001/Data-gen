import os, sys
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(here, ".env"))
from openai import OpenAI
mid = sys.argv[1] if len(sys.argv) > 1 else "gpt-5.6-luna"
c = OpenAI(api_key=os.environ["OPUSKEY"], base_url="https://api.opusgate.dev/v1")
try:
    r = c.chat.completions.create(model=mid, max_tokens=60, temperature=0.7,
                                  messages=[{"role": "user", "content": "Say hi in one short sentence."}])
    print("OK:", repr((r.choices[0].message.content or "")[:120]))
    print("USAGE:", r.usage)
except Exception as e:
    print("FAIL:", str(e)[:400])
