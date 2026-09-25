# Token and target trace
```json
{
  "state": "B is right of A.",
  "questions": {
    "spatial": {
      "type": "choice",
      "instructions": "B relative to A?",
      "criteria": {
        "left": "left",
        "right": "right"
      }
    }
  },
  "labels": {
    "spatial": "right"
  }
}
```
Target index: 1; candidates: ['left', 'right']
| Position | Token ID | Token | Role |
|---|---|---|---|
| 0 | 1 | <bos> | state |
| 1 | 4 | <state> | state |
| 2 | 47 | B | state |
| 3 | 272 | Ġis | state |
| 4 | 775 | Ġright | state |
| 5 | 239 | Ġof | state |
| 6 | 284 | ĠA | state |
| 7 | 27 | . | state |
| 8 | 5 | </state> | state |
| 9 | 6 | <q> | question branch |
| 10 | 12 | <choice> | question branch |
| 11 | 47 | B | question branch |
| 12 | 2313 | Ġrelative | question branch |
| 13 | 254 | Ġto | question branch |
| 14 | 284 | ĠA | question branch |
| 15 | 44 | ? | question branch |
| 16 | 8 | <opt> | question branch |
| 17 | 612 | left | question branch |
| 18 | 9 | </opt> | option endpoint |
| 19 | 8 | <opt> | question branch |
| 20 | 567 | right | question branch |
| 21 | 9 | </opt> | option endpoint |
| 22 | 10 | <decide> | decide |
| 23 | 7 | </q> | question branch |