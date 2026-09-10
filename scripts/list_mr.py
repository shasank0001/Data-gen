import os
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(here, ".env"))
from openai import OpenAI
mr = OpenAI(api_key=os.environ["MINIROUTER_KEY"], base_url="https://api.minirouter.sh/v1")
try:
    models = mr.models.list()
    ids = sorted(m.id for m in models)
    print("total models:", len(ids))
    for i in ids:
        if "luna" in i.lower() or "5.6" in i or "sol" in i.lower():
            print("HIT:", i)
except Exception as e:
    print("LIST ERR:", str(e)[:300])
