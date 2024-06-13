#!/usr/bin/env bash

pytest -v -s --cov=openmmopes --cov-report=term-missing --cov-report=html --pyargs --doctest-modules "$@" openmmopes
