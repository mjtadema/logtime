#!/usr/bin/env python3
# coding: utf-8
"""
Read a GROMACS log file for the checkpoint
timestamps to calculate the performance
"""
import re
import os
import typing
from typing import Union
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

def readlines_reversed(fh: typing.BinaryIO, bufsize=8192):
    """Generator that reads lines from a file in reverse by using a buffer.
    kinda stole this from https://stackoverflow.com/questions/2301789/how-to-read-a-file-in-reverse-order"""
    segment = None
    fh.seek(0, os.SEEK_END)  # 0 bytes from the end of the file
    filesize = remaining_size = fh.tell()
    offset = 0
    while remaining_size > 0:
        offset = min(filesize, offset + bufsize)
        fh.seek(filesize - offset)
        buffer = fh.read(min(remaining_size, bufsize)).decode(encoding='utf-8')
        remaining_size -= bufsize
        lines = buffer.split('\n')
        # chunks will be truncated
        if segment is not None:
            # If there was something left in segment, append it to the last line
            if buffer[-1] != '\n':
                lines[-1] += segment
            else:
                yield segment
        segment = lines.pop() # this should pop 0 by default?
        for line in reversed(lines):
            yield line
    if segment is not None:
        yield segment # yield the last segment as well

def parse_checkpoints(filename: Path, nsamples=4, **kwargs):
    cps = set()
    with open(filename, 'rb') as f:
        for line in readlines_reversed(f):
            match = re.match(re_checkpoint, line)
            if match:
                logger.debug(f"Matched line:\n{line}")
                cp = Checkpoint.from_regex(*match.groups())
                logger.debug(f"adding checkpoint {cp}.")
                cps.add(cp)
                logger.debug(f"{len(cps)} checkpoints of {nsamples}.")
                if len(cps) >= nsamples:
                    break
    return cps

def nsday(delta: Checkpoint, dt: float):
    # ns/day    steps      days          ps/step  ns
    return (delta.steps / delta.days) * (dt / 1000)

def parse_arguments() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", type=Path,
                        help="log file to read.")
    parser.add_argument("--verbose", '-v', action="store_true", default=False,
                        help="Show debug messages.")
    parser.add_argument("--hpu", action="store_true", default=False,
                        help="Show time in hour per us")
    parser.add_argument("--nsamples", '-n', type=int, default=4,
                        help="Number of checkpoint samples to take (default: 4).")
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    if not args.filename.exists():
        raise FileNotFoundError(f"{args.filename} not found.")

    return vars(args)

def hours_per_us(t: float) -> float:
    return (24 / t) * 1000

def parse_dt(filename: Union[Path,str], **kwargs):
    """Read the dt from logfile"""
    filename = Path(filename)
    with open(filename) as f:
        for line in f:
            if line.startswith("   dt"):
                dt = line.strip().split()[-1]
    return float(dt)

def main(*, filename: Union[Path,str], hpu: bool, **kwargs):
    filename = Path(filename)
    # Read the timestep value from input parameters
    dt = parse_dt(filename, **kwargs)
    # Simply subtract the last checkpoint from the second to last one
    cps = list(sorted(parse_checkpoints(filename, **kwargs)))
    if len(cps) < 2:
        e = ValueError(
            "Could not read enough checkpoints from", filename
        )
        logger.exception(e)
        raise e
    # zip(cps[:-1], cps[1:]) is taking the difference between neighboring checkpoints
    deltas = [after - before for before, after in zip(cps[:-1], cps[1:])]
    # Filter out whatever is >0.5 hours
    deltas = filter(lambda delta: delta / timedelta(hours=1) < 0.5, deltas)
    unit = "ns/day"
    times = [nsday(delta, dt) for delta in deltas]
    if hpu:
        unit = "hours/us"
        times = [hours_per_us(t) for t in times]
    last = cps[-1]
    lasttime = last.steps * (dt / 1000) # ns
    std = 0
    if len(cps) >= 3:
        std = stdev(times)
    logger.info("{}\t:\t{:.2f} ns\t@{:.2f} {} (±{:.2f})".format(
        filename, lasttime, mean(times), unit, std))

if __name__ == "__main__":
    kwargs = parse_arguments()
    main(**kwargs)
