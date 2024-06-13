#!/usr/bin/env bash

pytest -v -s --cov=openmm-opes --cov-report=term-missing --cov-report=html --pyargs --doctest-modules "$@" openmm-opes
