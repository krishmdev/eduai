# Curriculum taxonomies

There's one YAML file per subject, structured as subject → units (with blueprint weights) →
topics → learning objectives. Each objective has an id like `BIO.3.2.a`, a sentence, keywords, and
a Bloom level.

| File | Subject | Units | Objectives |
|---|---|---|---|
| `ap_biology.yaml` | AP Biology | 8 | 36 |
| `ap_chemistry.yaml` | AP Chemistry | 9 | 29 |
| `ap_physics_1.yaml` | AP Physics 1 | 9 | 28 |
| `ap_environmental_science.yaml` | AP Environmental Science | 9 | 30 |

- The scope loosely follows each AP course. Unit names are short generic titles, and every
  objective sentence was written for this project.
- The weights are our own rough blueprint numbers, not exam percentages.
- Physics 1 keeps a waves unit and a simple-circuits unit from the older course scope, because
  SciQ has many questions on both.

`uv run eduai curriculum validate` checks for unique ids, id prefixes, weights that sum to 1, known
Bloom levels, and at least three keywords per objective.

AP is a registered trademark of the College Board, which isn't affiliated with this project.
