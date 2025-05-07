Based on https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html

Use https://jupytext.readthedocs.io to create a notebook.

```sh
$ jupytext --to ipynb notebook.py
$ jupyter-notebook notebook.ipynb
```

For convenience the ipynb file is included. Please keep it in sync with the py
file using `jupytext` (see `convert-to-ipynb.sh`).

One-time setup of venv and ipy kernel:

```sh
# or
#   $ python -m venv --system-site-packages bayes-ml-course-sys
#   $ source ./bayes-ml-course-sys/bin/activate
$ mkvirtualenv --system-site-packages bayes-ml-course-sys
$ pip install -r requirements.txt

# Install custom kernel, select that in Jupyter. --sys-prefix installs into the
# current venv, while --user would install into ~/.local/share/jupyter/kernels/
$ python -m ipykernel install --name bayes-ml-course --sys-prefix
Installed kernelspec bayes-ml-course in /home/elcorto/.virtualenvs/bayes-ml-course-sys/share/jupyter/kernels/bayes-ml-course
```
