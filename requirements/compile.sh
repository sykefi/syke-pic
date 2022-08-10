#!/bin/sh

pip-compile --generate-hashes ./gpu.in &&
pip-compile --generate-hashes ./cpu.in &&
pip-compile --generate-hashes ./dev.in
