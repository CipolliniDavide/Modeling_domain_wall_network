# Networkqit Patch

A patch is necessary in order for networkqit to work with the dependencies listed in pyproject.toml (and specific versions in poetry.lock).
In `path/to/your/venv/lib/python3.11/site-packages/autograd/scipy/misc.py`, replace line 6 with:

`if not hasattr(osp_misc, 'logsumexp'):`.

[//]: # (In `autograd/scipy/misc.py`, replace line 6 with `if not hasattr&#40;osp_misc, 'logsumexp'&#41;:`.)
