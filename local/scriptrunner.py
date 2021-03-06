#AUTOGENERATED! DO NOT EDIT! File to edit: dev/0) scriptrunner.ipynb (unless otherwise specified).

__all__ = ['run_command']

#Cell
import subprocess
import shlex
import datetime
from fastai2.basics import Path
import json

#Cell
def _now(): return datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

def _add_dict_to_json(fn, d):
    "Adds a dictionary to json-like file or creates one"
    assert type(d) == dict
    path = Path(fn)
    if path.exists(): l = json.loads(path.open().read())
    else: l = []
    l.append(d)
    with open(fn, "w") as f: f.write(json.dumps(l))

def run_command(command, logfn=None):
    "Run shell command as an external process, optionally write logs to logfn"
    if type(command) == str: command = shlex.split(command)
    elif type(command) == list: command = command
    else: raise AssertionError("Command should be string or list")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = []
    start_time = _now()
    while True:
        output = process.stdout.readline()
        if output == b'' and process.poll() is not None: break
        if output:
            _out = output.decode(); print(_out.strip())
            stdout.append(_out)
    end_time = _now()
    rc = process.poll()
    _, stderr =  process.communicate()
    err = stderr.decode(); print(err)
    out = "".join(stdout)
    if logfn:
        d = {"start_time": start_time, "end_time": end_time,
             "command": command, "stderr":err, "stdout":out}
        _add_dict_to_json(logfn, d)
    return rc