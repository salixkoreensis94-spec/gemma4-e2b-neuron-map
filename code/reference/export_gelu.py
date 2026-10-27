"""Export reference CPU GELU table from a caller-supplied shared library."""
import argparse,ctypes,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
lib=ctypes.CDLL(str(a.library.resolve()));lib.ggml_cpu_init()
data=bytes((ctypes.c_uint16*65536).in_dll(lib,'ggml_table_gelu_f16'))
with a.output.open('xb') as f:f.write(data)
print(json.dumps(dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())))
