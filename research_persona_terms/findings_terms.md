# Persona terms: concise research findings

## Definitions

- **Skepticism** is an attitude of doubt and, more importantly for this task, a method of suspended judgment, systematic doubt, or criticism. Merriam-Webster says the term implies unwillingness to believe without conclusive evidence. It is therefore an *evidence-calibration stance*, not a synonym for knowing the factual answer.  
  Source: [Merriam-Webster, “skepticism”](https://www.merriam-webster.com/dictionary/skepticism)

- **Credulity** (and *credulousness*) is willingness/readiness to believe something, especially when it is unlikely or supported by slight or uncertain evidence. A **credulous** person has a low threshold for accepting a claim.  
  Sources: [Merriam-Webster, “credulity”](https://www.merriam-webster.com/dictionary/credulity), [Cambridge, “credulity”](https://dictionary.cambridge.org/us/dictionary/english/credulity), [Merriam-Webster, “credulous”](https://www.merriam-webster.com/dictionary/credulous)

- **Gullibility** overlaps with credulity, but foregrounds being easily deceived, duped, or cheated—especially by believing an untrue claim. It concerns susceptibility to manipulation/deception, not merely a general willingness to believe.  
  Sources: [Merriam-Webster, “gullibility”](https://www.merriam-webster.com/dictionary/gullibility), [Merriam-Webster, “gullible”](https://www.merriam-webster.com/dictionary/gullible), [Cambridge, “gullible”](https://dictionary.cambridge.org/us/dictionary/english/gullible)

## Implications for the current draft

1. Do **not** call a skeptical system prompt a direct expression of the model’s “world-model.” It should request evidence checking, calibrated uncertainty, and withholding belief until the evidence is adequate. It can still be wrong, overly doubtful, or persuaded by poor evidence.
2. Do **not** treat the *Credulous persona* and a *gullible human* as the same experimental condition. A credulous persona is an assistant instructed to accept claims too readily; a gullible human is a user vulnerable to accepting deceptive claims or advice.
3. The requested P/Q contrast should be defined by **user vulnerability**, not by asking the assistant to mimic it. For P (gullible-user condition), an appropriate assistant remains truthful, checks questionable claims, makes risks salient, and avoids exploitative or overconfident persuasion. For Q (non-gullible-user condition), it can be equally truthful but may assume basic verification literacy. The answer must not become false merely because the user is gullible.
4. Keep the two axes explicit in the schema: `assistant_persona` (`skeptical`, `credulous`, or neutral) versus `user_susceptibility` (`gullible`, `non_gullible`). Otherwise labels will confound whose belief/behaviour is being varied.

## Suggested one-sentence definitions for the dataset spec

- **Skeptical assistant:** evaluates claims against available evidence, asks for support when needed, and states uncertainty rather than accepting unsupported claims.
- **Credulous assistant:** tends to accept claims on weak evidence and is easily led by assertions; use only when intentionally studying that failure mode.
- **Gullible user:** readily trusts dubious claims or offers and may be susceptible to deception; the assistant should respond with added clarity and protective guidance, not exploit that vulnerability.
