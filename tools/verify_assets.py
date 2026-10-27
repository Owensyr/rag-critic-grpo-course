"""Verify a previously unpacked private bundle; no automatic tar extraction."""
import argparse
from pathlib import Path
from common import abspath,load,sha
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',default='private');a=p.parse_args();root=abspath(a.directory).resolve()
    manifest=load(root/'assets_manifest.json')
    for name,item in manifest['files'].items():
        file=(root/name).resolve()
        if not file.is_relative_to(root):raise ValueError('Manifest path escapes bundle')
        if file.stat().st_size!=item['bytes'] or sha(file)!=item['sha256']:raise ValueError('Asset mismatch: '+name)
    print('PRIVATE_ASSETS_OK',len(manifest['files']))
if __name__=='__main__':main()
