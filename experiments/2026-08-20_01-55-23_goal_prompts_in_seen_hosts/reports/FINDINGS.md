# Findings (written 2026-08-20 after 9/9 points, determinism smoke PASSED)

| host (seen control) | goal prompt | prompted | host-skill fallback |
|---|---|---:|---:|
| close top drawer (20/20) | open the middle drawer... | 0/20 | **19/20** |
| white bowl on cabinet (15/20) | put the bowl on the stove | 0/20 | 0/20 |
| wine bottle on rack (8/20) | put the wine bottle on top of the cabinet | 0/20 | 0/20 |

- **R1gh fires** (on the 2 healthy hosts; the wine host fails R4gh's 0.5
  control bar and is excluded from rule counting): verbatim goal strings
  retrieve nothing even in scenes with compatible objects — including
  goals 1/2, unhostable in v2 and semantically remapped here.
- **R3gh does NOT fire**: host-skill fallback under a novel prompt happened
  only on the drawer host (19/20; v2 saw 15/20 under stream noise) and not
  on the bowl/wine hosts (0/20). v2's "graded fallback" is therefore NOT
  general — it appears specific to prompts sharing the "...the X drawer of
  the cabinet" frame with the host instruction.
- Together with the k=0 probes: goal instructions cannot be unlocked at
  zero-shot from the language side in ANY scene we tried.
