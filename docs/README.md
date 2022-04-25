# Generating documentation

Neural Lifetimes uses Sphinx to auto generate its own documentation. This document provides a guide to update our documentation with your modifications.

You can also check out [Sphinx's documentation](https://www.sphinx-doc.org/en/master/usage/quickstart.html)  to see how to [write the rst files])(https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html), and [document code](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#module-sphinx.ext.autodoc).


## Incorporating notebooks

Sphinx on its own doesn't allow us to incorporate notebooks into the documentation easily. Here is how to do so with the [nbsphinx](https://github.com/spatialaudio/nbsphinx) and [nbsphinx-link](https://github.com/vidartf/nbsphinx-link) libraries (which are included in this package):

1. Edit `index.rst` and add the names of your jupyter notebooks (without ".ipynb") to the toctree.
2. For notebooks which are not in the docs folder, instead create a `.nblink` file that links to it like so
```
    {
        "path": "/relative/path/to/notebook.ipynb",
        "extra-media": [
            "/relative/path/to/images/folder",
            "/relative/path/to/specific/image.png"
        ]
    }
```
where the path is the relative path to the `docs` folder.
3. Run `make html` to build the html files.

To link to the documentation of other functions, use `:mod:|bt|DisplayName <module.class.targetName>|bt|`.
