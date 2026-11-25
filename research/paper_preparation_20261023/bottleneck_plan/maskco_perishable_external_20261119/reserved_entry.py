"""Run the next locked recipe against the same independent benchmark harness."""
import sys
from types import SimpleNamespace
import external_common as common

def main():
    if len(sys.argv)==3 and sys.argv[2] in ("HYBRID","HYBRID_D"):
        import previous_pool
        sys.modules["pool_search"]=previous_pool
        original_read=common.read
        def read(path):
            v=original_read(path)
            if path.name=="locked_method.json":v=dict(v,candidate="HYBRID",matched="HYBRID_D")
            return v
        common.read=read
    if len(sys.argv)==3 and sys.argv[2]=="STATIC":
        original=common.study.load
        def load(key,access,core,source,locked,case):
            from encoded_mask import load_split
            return load_split(SimpleNamespace(load=original),access,core,source,locked,case)
        common.study.load=load
    common.main()
if __name__=="__main__":main()
