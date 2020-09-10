# -*- coding: utf-8 -*-
import hashlib
import json
import os

import joblib


def md5(data):
    if isinstance(data, str):
        data = data.encode("UTF-8")

    return hashlib.md5(data).hexdigest()


def make_dirs(dirname):
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def read_file(filename):
    with open(filename, mode="rb") as f:
        return f.read()


def write_file(data, filename):
    make_dirs(os.path.dirname(filename))

    with open(filename, mode="wb") as f:
        f.write(data)


def read_text(filename, encoding="UTF-8"):
    with open(filename, mode="r", encoding=encoding) as f:
        return f.read()


def write_text(text, filename, encoding="UTF-8"):
    make_dirs(os.path.dirname(filename))

    with open(filename, mode="w", encoding=encoding) as f:
        f.write(text)


def read_lines(filename, encoding="UTF-8"):
    with open(filename, mode="r", encoding=encoding) as f:
        for line in f:
            yield line.rstrip("\r\n\v")


def write_lines(lines, filename, linesep="\n", encoding="UTF-8"):
    make_dirs(os.path.dirname(filename))

    with open(filename, mode="w", encoding=encoding) as f:
        for line in lines:
            f.write(line)
            f.write(linesep)


def read_json(filename, encoding="UTF-8"):
    with open(filename, mode="r", encoding=encoding) as f:
        return json.load(f)


def write_json(obj, filename, indent=None, encoding="UTF-8"):
    make_dirs(os.path.dirname(filename))

    with open(filename, mode="w", encoding=encoding) as f:
        json.dump(obj, fp=f, ensure_ascii=False, indent=indent)


def read_json_lines(filename, encoding="UTF-8"):
    for json_line in read_lines(filename, encoding=encoding):
        yield json.loads(json_line)


def write_json_lines(lines, filename, linesep="\n", encoding="UTF-8"):
    json_lines = (json.dumps(line, ensure_ascii=False) for line in lines)
    write_lines(json_lines, filename=filename, linesep=linesep, encoding=encoding)


def deserialize(filename, mmap_mode=None):
    return joblib.load(filename, mmap_mode=mmap_mode)


def serialize(value, filename, compress=3, protocol=None, cache_size=None):
    make_dirs(os.path.dirname(filename))

    joblib.dump(
        value,
        filename=filename,
        compress=compress,
        protocol=protocol,
        cache_size=cache_size,
    )
