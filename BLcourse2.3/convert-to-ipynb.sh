#!/bin/sh

# Remove plots etc, but keep random cell IDs. We need this b/c we use jupytext
# --update below.
nbstripout --keep-id *.ipynb

for fn in 0*dim.py; do
    # --update keeps call IDs -> smaller diffs
    jupytext --to ipynb --update $fn
done
