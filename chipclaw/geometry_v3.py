"""Reuse frozen, hash-checked physical pages; isolate any newly extracted pages."""
import hashlib,json
from .config import ROOT
from .geometry import GeometryLoader

class V3GeometryLoader(GeometryLoader):
    def __init__(self):
        super().__init__(ROOT/'.cache/v3/pdf_geometry')
        self.folder=ROOT/'data/corpus-v3/geometry';self.loaded={}
        manifest=json.loads((self.folder/'manifest.json').read_text())
        self.expected={r['doc_id']:r['sha256'] for r in manifest['files']}

    def read_page(self,path,page,digest):
        doc=path.stem
        if doc in self.expected:
            if doc not in self.loaded:
                source=self.folder/(doc+'.json')
                if hashlib.sha256(source.read_bytes()).hexdigest()!=self.expected[doc]:raise ValueError('Frozen physical page fingerprint mismatch')
                payload=json.loads(source.read_text(encoding='utf-8'))
                if payload['pdf_sha256']!=digest:raise ValueError('Physical source PDF fingerprint mismatch')
                runtime=super().identity()['runtime']
                if payload['pdfplumber']!=runtime['pdfplumber'] or payload['pdfminer']!=runtime['pdfminer']:
                    raise ValueError('Frozen physical extraction dependency identity mismatch')
                self.loaded[doc]=payload
            if str(page) in self.loaded[doc]['pages']:
                self.cache_hits+=1
                return self.loaded[doc]['pages'][str(page)]
        return super().read_page(path,page,digest)

    def identity(self):
        return {**super().identity(),'frozen_geometry_manifest_sha256':hashlib.sha256((self.folder/'manifest.json').read_bytes()).hexdigest(),
                'scope':'240 frozen source pages; other pages extracted in isolated v3 cache'}
