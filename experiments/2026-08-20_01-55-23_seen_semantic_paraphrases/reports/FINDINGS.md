# Findings (written 2026-08-20 after 20/20 points, determinism smoke PASSED)

Mean delta -0.36; McNemar p<0.05 on 4/10 pairs. Neither R1sp (robust) nor
R2sp (uniform collapse) fires -> **R3sp: brittleness depends on the edit
type**, despite every paraphrase preserving all task-defining descriptors:

| edit type | pairs | deltas |
|---|---|---|
| verb swap (put->place) | bowl-on-plate, wine-rack, frying-pan | -0.35, -0.25, 0.00 |
| verb+particle reorder ("switch the stove on") | stove | -0.05 |
| possessive restructure ("cabinet's top drawer") | 2 pairs | **-0.55, -0.65** (both p<0.001) |
| rare token ("atop") | bowl-on-cabinet | **-0.95** (p=4e-6) |
| compartment reorder ("caddy's left compartment") | book | **-0.60** (p=5e-4) |
| additive ("microwave door"), comma insert | 2 pairs | -0.10, -0.10 |

- Descriptor preservation does NOT guarantee robustness: structural
  rewrites (possessives, reordering) and out-of-distribution tokens break
  retrieval even with all content words intact.
- Refines v1/v2: the collapse there was not ONLY descriptor loss; the
  selector is sensitive to phrase structure. Relabeling coverage must
  include structural variants, not just synonym swaps.
