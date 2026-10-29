"""One estimator per error term, each with its own data requirement.

A term is only as good as the geometry that constrains it, so every estimator
states what it needs before it solves and says which direction is missing when
it cannot.
"""
from __future__ import annotations
