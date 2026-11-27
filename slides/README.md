# Presentation slides

The deck is one HTML file, [`index.html`](index.html). It needs no install or internet connection.

## View it

Open `slides/index.html` in any browser (double-click it).

| Key | Action |
|---|---|
| → / Space / click right side | Next slide |
| ← / click left side | Previous slide |
| F | Fullscreen |

To export a PDF: Ctrl+P → "Save as PDF", margins "None", background graphics on.

## Who edits what

| Section | Presenter | Tasks | Read first |
|---|---|---|---|
| Title, agenda, Tasks 1–2 | Naaz | Problem, dataset, preprocessing | `docs/task1_*.md`, `docs/task2_*.md` |
| Tasks 3–4 | Halas | Model, training (+ Task 9 if profiling is ready) | `docs/task3_*.md`, `docs/task4_*.md` |
| Tasks 5–6 | Ashley | TensorBoard curves, evaluation | `docs/task5_*.md`, `docs/task6_*.md`, `docs/figures/task6/summary.md` |
| Tasks 7–8, wrap-up | Chris | Distributed training, scaling | `docs/task7_*.md`, `docs/task8_*.md` |

Each section in `index.html` is between `===== BEGIN <NAME>` and `===== END <NAME>` comments. **Only edit inside your own section.** Then git can merge everyone's changes without conflicts.

The slides already have the main numbers and figures. Your job is to:

1. Fill in or delete everything highlighted **yellow** (`class="todo"`).
2. Keep the numbers exactly as they appear in the docs, so all our slides match.

Aim for about 3–4 minutes per person, 15 minutes in total.

## Editing tips

- Copy an existing slide (`<section class="slide">…</section>`) to make a new one. Set `data-presenter="YourName"`.
- Layouts: `.cols` (two columns), `.stats` (big-number boxes), `<table>`, `<figure><img></figure>`.
- Figures live in `docs/figures/`. Reference them as `../docs/figures/task6/confusion_top.png`.
- Add `data-hidden` to a `<section>` to skip it without deleting it.
- Refresh the browser after saving. The URL ends in `#5` etc., so you stay on the same slide.

## Git

Same workflow as [`docs/TEAMMATE_GUIDE.md`](../docs/TEAMMATE_GUIDE.md) §4: branch off `main`, commit your section, push, open a PR.

```bash
git checkout main && git pull
git checkout -b <you>/slides
# edit your section of slides/index.html
git add slides/index.html
git commit -m "Slides: <your section>"
git push -u origin <you>/slides
```
