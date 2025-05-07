#!/bin/sh

for fn in 0*dim.py; do
    jupytext --to ipynb --update $fn
done
