"""Additive safe copies of the 18 previously published V3/V6 archive paths."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.export_public_shore_bess_v8 import export, scan_bytes, scan_serialized
from app.services.rl_model.shore_bess.v8_public_artifacts import verify_model_export
DIRECTORY = 'evidence/public_models/legacy_v3_v6_20260912'
MANIFEST = DIRECTORY + '/manifest.json'
SCHEMA = 'port-sb3-legacy-public-metadata-export.v1'


def verify(root=ROOT):
    import zipfile
    manifest=json.loads((root/MANIFEST).read_text())
    if manifest['schema']!=SCHEMA or manifest['source_model_archives_modified'] is not False or manifest['historical_reports_modified'] is not False:
        raise ValueError('legacy model export manifest differs')
    for row in manifest['models']:
        path=verify_model_export(root,row,export_directory=DIRECTORY)
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None:raise ValueError('public archive CRC failed')
            for name in z.namelist():scan_bytes(z.read(name),name)
            scan_serialized(json.loads(z.read('data')))
    return {'status':'PASS','models':len(manifest['models']),
            'original_archive_paths':sum(len(r['source_paths']) for r in manifest['models']),
            'historical_reports_modified':False,'production_authority':False}


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify',action='store_true');args=p.parse_args()
    if not args.verify:
        inventory=json.loads((ROOT/DIRECTORY/'original_archives.json').read_text())
        sources={r['path']:r['algorithm'] for r in inventory['archives']}
        from app.services.rl_model.shore_bess.v8_public_artifacts import sha256
        for r in inventory['archives']:
            if sha256(ROOT/r['path'])!=r['sha256']:raise ValueError('original legacy archive changed')
        export(sources,DIRECTORY,MANIFEST,SCHEMA)
    print(json.dumps(verify()))
if __name__=='__main__':main()
