# Lecture slides

This folder is where the lecture slides go after you download them.

**The slides are for participants of this course only.** Do not upload, share, or commit them, or
anything derived from them (extracted text, screenshots, …). This includes your fork of this
repository, which is probably public. Git ignores everything in this folder except the three
files that belong to the repository. Please don't work around that, e.g. with `git add -f`.

```bash
python slides/download.py           # download all available slides / update changed ones
python slides/download.py 6 7       # only lectures 6 and 7
python slides/download.py --force   # download everything again
```

- You need the course login announced in the lecture. The script asks for it, or reads it from
  the environment variables `CV_SLIDES_USER` and `CV_SLIDES_PASSWORD`. Don't save the login in
  files inside the repository.
- The script only downloads files that are new or have changed on the website. Slides are sometimes
  updated during the semester, so re-running it is cheap.
- The recordings are linked on the [course website](https://lmb.informatik.uni-freiburg.de/lectures/computer_vision_I/).
