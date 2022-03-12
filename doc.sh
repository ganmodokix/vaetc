#!/bin/bash

sphinx-apidoc -f -o ./docs .
sphinx-build ./docs ./docs/_build