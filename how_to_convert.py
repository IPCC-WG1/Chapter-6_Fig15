# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## How to convert to pdf:

# %% [markdown]
# 1. Have all neccessary latex packages
# 2. Have the file [citations.tplx](citations.tplx) in same folder (gives the template for latex) and your bib file (here IPCC_terje.bib)
# 3. Edit the notebook metadata, e.g.:
# ```
#  "latex_metadata": {
#         "author": "Sara Blichner, T. K. Berntsen",
#         "bibliography": "IPCC_terje",
#         "bibstyle": "plain",
#         "title": "Comparison of temperature response for various climate gases"
#     }
# ```
#
#

# %% [markdown]
# 4. Edit figure captions with cell metadata:
# ```
# {
#     "caption": "some caption",
#     "label": "fig:some",
# }
# ```

# %% [markdown]
# 5. Run script:
# ```bash
#  ./convert_notebook.sh notebook_name citations.tplx 
#
# ```
# (Notebook name without extension)
