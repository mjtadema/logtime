#!/usr/bin/env python3
# coding: utf-8
"""
Read a GROMACS log file for the checkpoint
timestamps to calculate the performance
"""
import re
from collections import namedtuple
from datetime import datetime, timedelta
import calendar
from pathlib import Path
import sys
from statistics import mean, stdev
import argparse

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

re_checkpoint = re.compile(
    r'^Writing checkpoint, step (\d+) at \w{3} (\w{3})\W{1,2}(\d{1,2}) (\d{1,2}):(\d{1,2}):(\d{1,2}) (\d{4})$'
)

month_to_i = {month.lower():i for i, month in enumerate(calendar.month_abbr)}

CpParse = namedtuple('CpParse', ['steps','M','D','h','m','s','Y'])

class Checkpoint:
    def __init__(self, steps: int, timestamp: datetime):
        self.steps = steps
        self.timestamp = timestamp
    def __sub__(self, other):
        # Return a new cp object with the delta
        step = self.steps - other.steps
        timedelta = self.timestamp - other.timestamp
        return Checkpoint(step, timedelta)
    def __repr__(self):
        return f"{self.__class__.__name__} at {self.steps} ({self.timestamp})"
    def __hash__(self):
        return hash(self.steps)
    def __eq__(self, other):
        return self.steps == other.steps
    def __lt__(self, other):
        return self.steps < other.steps
    def __gt__(self, other):
        return self.steps > other.steps
    def __truediv__(self, other):
        return self.timestamp / other
    @property
    def days(self):
        return self / timedelta(days=1)
    @classmethod
    def from_regex(cls, *args):
        parsed = CpParse(*args)
        timestamp = datetime(
            int(parsed.Y), month_to_i[parsed.M.lower()], int(parsed.D),
            int(parsed.h), int(parsed.m), int(parsed.s)
        )
        return cls(int(parsed.steps), timestamp)

def parse_checkpoints(filename: Path, buffsize=(2**12)):
    cps = set()
    i = 1
    with open(filename, 'rb') as f:
        for line in readlines_reversed(f):
            match = re.match(re_checkpoint, line)
            if match:
                cp = Checkpoint.from_regex(*match.groups())
                logger.debug(f"adding checkpoint {cp}.")
                cps.add(cp)
                logger.debug(f"{len(cps)} checkpoints of {nsamples}.")
                if len(cps) >= nsamples:
                    break
    return cps

def nsday(delta: Checkpoint):
    # ns/day    steps      days          ps/step  ns
    return (delta.steps / delta.days) * (0.002 / 1000)

def main(filename):
    # Simply subtract the last checkpoint from the second to last one
    cps = list(sorted(parse_checkpoints(filename)))
    assert len(cps) > 1, print("Could not read checkpoints from", filename, file=sys.stderr)
    # zip(cps[:-1], cps[1:]) is taking the difference between neighboring checkpoints
    deltas = [after - before for before, after in zip(cps[:-1], cps[1:])]
    # Filter out whatever is >0.5 hours
    deltas = filter(lambda dt: dt / timedelta(hours=1) < 0.5, deltas)
    times = [nsday(dt) for dt in deltas]
    last = cps[-1]
    lasttime = last.steps * (0.002 / 1000)
    print(f"{filename}\t:\t{lasttime:.2f} ns\t@{mean(times):.2f} ns/day (±{stdev(times):.2f})")

if __name__ == "__main__":
    kwargs = parse_arguments()
    main(**kwargs)
