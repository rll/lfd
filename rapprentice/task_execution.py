"""
Misc functions that are useful in the top-level task-execution scripts
"""


def request_int_in_range(too_high_val):
    while True:
        try:
            choice_ind = int(raw_input())
        except ValueError:            
            pass
        if choice_ind <= too_high_val:
            return choice_ind
        print "invalid selection. try again"


from datetime import datetime
class ExecutionLogEntry(object):
    def __init__(self, step, name, data, description=""):
        self.time = datetime.now().isoformat()
        self.step, self.name, self.data, self.description = step, name, data, description

# class Log(object):
#     def __init__(self, filename, max_unwritten=10):
#         self.filename = filename
#         self.hdf = h5py.File(filename, "w")
#         self.unwritten_buf = []
#         self.num_written = 0
#         self.max_unwritten = max_unwritten

#     def append(self, entry):
#         self.unwritten_buf.append(entry)
#         if len(self.unwritten_buf) > self.max_unwritten:
#           self.flush()

#     def __call__(self, *args, **kwargs):
#         self.append(LogEntry(*args, **kwargs))

#     def _write_entry(self, entry, group_name):
#         group = self.hdf.create_group(group_name)
#         for k, v in entry.__dict__.items():
#             group[k] = v

#     def flush(self):
#         for entry in self.unwritten_buf:
#             self._write_entry(entry, "%015d" % self.num_written)
#             self.num_written += 1
#         self.hdf.flush()
#         self.unwritten_buf = []

#     def close(self):
#         self.flush()
#         self.hdf.close()
import cPickle
class ExecutionLog(object):
    def __init__(self, filename, max_unwritten=10):
        self.filename = filename
        self.entries = []
        self.max_unwritten = max_unwritten
        self.num_unwritten = 0

    def append(self, entry):
        self.entries.append(entry)
        self.num_unwritten += 1
        if self.num_unwritten > self.max_unwritten:
            self.flush()

    def __call__(self, *args, **kwargs):
        self.append(ExecutionLogEntry(*args, **kwargs))

    def flush(self):
        with open(self.filename, "w") as f:
            cPickle.dump(self.entries, f, protocol=cPickle.HIGHEST_PROTOCOL)
        self.num_unwritten = 0

    def close(self):
        self.flush()
