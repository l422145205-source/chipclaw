"""Question-independent, fingerprinted source extraction using PDF rectangles."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from .config import ROOT
from .geometry_worker import VERSION

def worker_python():
    if os.environ.get('CHIPCLAW_PDF_PYTHON'):
        return Path(os.environ['CHIPCLAW_PDF_PYTHON'])
    if importlib.util.find_spec('pdfplumber'):
        return Path(sys.executable)
    bundled=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
    return bundled if bundled.is_file() else None

class GeometryLoader:
    def __init__(self, cache_root=None):
        self.cache_root=Path(cache_root or ROOT/'.cache/pdf_geometry')
        self.python=worker_python()
        self.worker=Path(__file__).with_name('geometry_worker.py')
        self.worker_sha=hashlib.sha256(self.worker.read_bytes()).hexdigest()
        self.runtime=None
        self.cache_hits=0
        self.cache_misses=0
    def identity(self):
        if self.runtime is None and self.python:
            code="import json,sys,pdfplumber,importlib.metadata as m;print(json.dumps({'python':sys.version,'pdfplumber':pdfplumber.__version__,'pdfminer':m.version('pdfminer.six')}))"
            result=subprocess.run([str(self.python),'-B','-c',code],capture_output=True,text=True,check=True,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            self.runtime=json.loads(result.stdout)
        return {'runtime':self.runtime,'worker_sha256':self.worker_sha,'version':VERSION}
    def read_page(self,path,page,digest):
        if not self.python:return None
        identity=self.identity()
        directory=self.cache_root/digest/self.worker_sha[:16]
        directory.mkdir(parents=True,exist_ok=True)
        target=directory/(str(page)+'.json')
        if target.exists():
            payload=json.loads(target.read_text(encoding='utf-8'))
            self.cache_hits+=1
        else:
            result=subprocess.run([str(self.python),'-B',str(self.worker),'--pdf',str(path),'--pages',json.dumps([page]),
                '--out',str(target)],capture_output=True,text=True,timeout=120,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            if result.returncode:
                raise RuntimeError('PDF geometry extraction failed: '+result.stderr[-600:])
            payload=json.loads(target.read_text(encoding='utf-8'))
            self.cache_misses+=1
        if payload['pdf_sha256']!=digest or payload['worker_sha256']!=self.worker_sha or payload['version']!=VERSION:
            raise ValueError('PDF geometry cache fingerprint mismatch')
        if payload['pdfplumber_version']!=identity['runtime']['pdfplumber'] or payload['pdfminer_version']!=identity['runtime']['pdfminer']:
            raise ValueError('Geometry dependency identity differs from source cache')
        return payload['pages'][str(page)]
