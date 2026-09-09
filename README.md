# Data-gen — toy gullibility dialogues

1 chat per call. High / Low / Medium, label from behavior only.

## Layout
```
prompts/high_v2.txt   # high behavior
prompts/low_v2.txt    # low behavior
prompts/medium_v2.txt # neutral baseline
scripts/gen.py        # ollama runner + validator
output/               # .txt + .json (prompt saved in json)
```

## Run (local Ollama)
```bash
pip install -r requirements.txt
ollama pull gemma4:e2b-it-qat
python scripts/gen.py --limit 6 --model gemma4:e2b-it-qat
```

## Rules enforced in code
- 6-10 turns, start user, alternate
- ban `gullible/gullibility` in chat
- ban quiz-meta (`the question asks...`)
- high must not ask for source, low must push back
```
