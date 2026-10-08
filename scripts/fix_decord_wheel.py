#!/usr/bin/env python3
"""Repair decord 0.6.0's incorrect internal tag to match its published filename."""
import base64
import csv
import hashlib
import importlib.metadata
import io
from pathlib import Path
import decord

assert decord.__version__ == '0.6.0'
dist = importlib.metadata.distribution('decord')
wheel = Path(dist.locate_file('decord-0.6.0.dist-info/WHEEL'))
record = wheel.with_name('RECORD')
original = wheel.read_text()
wrong = 'Tag: cp36-cp36m-manylinux2010_x86_64'
correct = 'Tag: py3-none-manylinux2010_x86_64'
if wrong in original:
    data = original.replace(wrong, correct).encode()
    wheel.write_bytes(data)
    rows = list(csv.reader(io.StringIO(record.read_text())))
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
    for row in rows:
        if row[0] == 'decord-0.6.0.dist-info/WHEEL':
            row[1:] = ['sha256=' + digest, str(len(data))]
    output = io.StringIO()
    csv.writer(output, lineterminator='\n').writerows(rows)
    record.write_text(output.getvalue())
    print('Repaired decord wheel metadata; package code and binaries unchanged.')
else:
    assert correct in original, 'Unexpected decord tag; refusing to patch'
    print('decord wheel metadata already matches the published py3-none wheel.')
