# Computer Vision — Exercises (WS 2026/27)

Exercise material for the **Computer Vision WS26/27** lecture by Prof. Thomas Brox
([Pattern Recognition and Image Processing group](https://lmb.informatik.uni-freiburg.de), University of Freiburg).

- **Course website** (schedule, slides, recordings, exam info):
  https://lmb.informatik.uni-freiburg.de/lectures/computer_vision_I/
- **Questions:** ask in the exercise sessions or in the
  [ILIAS forum](https://ilias.uni-freiburg.de/goto.php?target=crs_1798330&client_id=unifreiburg),
  please not by email.
- **Exercise sessions:** Mondays 16:00, room 101-01-016/18.

The exercises are **not graded**. They are there to help you understand the lecture by building
things yourself. The written exam (16.2.2027) is based on the lecture content, but it is much
easier to understand the concepts once you have implemented them.

## Getting started

1. Follow the setup instructions in [`setup.pdf`](setup.pdf). They cover Python 3.10 with conda,
   the TF pool, SSH, and GPUs.
2. To use the TF pool computers and datasets, register with the
   [TF pool access form](https://uimaa6oy.forms.app/uni-freiburg-tf-pool-access-updated).
3. Clone this repository. On the pool, clone it into `/project/cv-ws2627/<username>/`, because
   your home folder only has 1 GB.
4. Create and activate your environment, then install the requirements:
   `conda activate cvenv && pip install -U -r requirements.txt`
5. If something does not work, check [`FAQ.md`](FAQ.md) first.

## How the exercises work

Each week a new exercise is introduced in the session and pushed here. The solution follows one
week later. Run `git pull` to get new material.

```
exNN/
├── assignment_NN.pdf   # exercise sheet: read this first
├── code/               # your starting point: fill in the TODOs
├── solution/           # added one week after the exercise is introduced
└── solution_NN.pdf     # solution sheet (for some exercises)
```

- Fill in the parts of the code marked `# START TODO` / `# END TODO`. They currently
  `raise NotImplementedError`.
- Run all commands from inside `exNN/code/`. If an exercise contains a `tests/` folder,
  check your solution with `python -m pytest`.
- Some exercises use Jupyter notebooks. Start them with `jupyter lab`.

## Schedule

| Date | Exercise session | Topic of the new exercise |
|---|---|---|
| 26.10. | Intro Exercise 01 | Linear algebra & NumPy refresher, diffusion filters |
| 2.11. | Solution 01 + Intro Exercise 02 | Deep learning 101: training a CNN on CIFAR-10 |
| 9.11. | Solution 02 + Intro Exercise 03 | Implementing convolutions, CNNs |
| 16.11. | *no session* | |
| 23.11. | Solution 03 + Intro Exercise 04 | Batch normalization and distribution shift |
| 30.11. | Solution 04 + Intro Exercise 05 | Optical flow with deep learning (FlowNet) |
| 7.12. | *no session* | |
| 14.12. | Solution 05 + Intro Exercise 06 | Camera calibration |
| 21.12. | *no session* | |
| | *winter break* | |
| 11.1. | Solution 06 + Intro Exercise 07 | Disparity estimation (DispNet) |
| later | to be announced | |

## Lecture slides

The [`slides/`](slides/) folder is for the lecture slides. To download them with the course login
announced in the lecture, run:

```bash
python slides/download.py
```

**The slides are for participants of this course only.** Do not upload, share, or commit them,
or material copied from them. This includes your fork of this repository, which is probably
public. Git ignores the downloaded files automatically. See
[`slides/README.md`](slides/README.md) for details.

## Using AI assistants

You are welcome to use AI coding assistants (Claude Code, Codex, Copilot, Cursor, Gemini, …) for
the exercises. [`AGENTS.md`](AGENTS.md) contains default instructions that most assistants pick up
automatically. They make the assistant act like a tutor: it explains concepts, points you to the
relevant lecture slides, and gives hints, but it does not just fill in the TODOs for you. The
assistant can also help with setup and technical problems.

You can change or ignore these instructions. But since the exercises are not graded, letting an AI
solve them only costs you the learning. If you use Gemini CLI, set `"contextFileName": "AGENTS.md"`
in its settings so it reads the file.
