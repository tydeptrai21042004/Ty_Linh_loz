from __future__ import annotations
import json
import zipfile
from pathlib import Path
import pytest
import yaml
from tests.multidataset_fixtures import create_dronevehicle, create_m3fd, create_vedai
from ultralytics.data.public_multidataset import (
    ADDITIONAL_DATASETS, _prepare_m3fd, _dronevehicle_polygon,
    _download_hf_zip_files, _extract_all_archives,
)
from paper_three_datasets import run


def test_three_datasets_have_paper_dois():
    from ultralytics.data.public_rgb_ir import get_public_dataset
    for name in ('vedai','dronevehicle','m3fd'):
        spec=get_public_dataset(name)
        assert spec.citation_doi.startswith('10.')
        assert spec.download_url


def test_dronevehicle_crop_and_native_polygon(tmp_path):
    from ultralytics.data.public_rgb_ir import prepare_public_dataset
    from ultralytics.data.rgb_ir_check import validate_dataset
    source=create_dronevehicle(tmp_path/'src')
    result=prepare_public_dataset('dronevehicle',tmp_path/'out',archive=source,accept_license=True)
    data=yaml.safe_load(result.read_text())
    assert 'test' in data
    assert data['names'][4]=='freight car'
    from PIL import Image
    img=next((result.parent/'rgb/images/test').rglob('*.jpg'))
    assert Image.open(img).size == (640,512)
    lab=next((result.parent/'labels/test').rglob('*.txt'))
    vals=[float(v) for v in lab.read_text().split()]
    assert len(vals)==9
    assert max(vals[1::2])<0.5 and max(vals[2::2])<0.5
    report=validate_dataset(str(result),'obb',fingerprint='metadata')
    assert report['ok'], report['errors']
    meta=json.loads((result.parent/'SOURCE_MANIFEST.json').read_text())
    assert meta['annotation_geometry']=='native_obb'
    assert meta['statistics']['splits']=={'train':2,'val':2,'test':2}


def test_m3fd_separate_zip_layout(tmp_path):
    root=create_m3fd(tmp_path/'src',pairs=6)
    src=root/'M3FD';zips=tmp_path/'zips';zips.mkdir()
    for name in ('vi','ir','labels'):
        with zipfile.ZipFile(zips/f'{name}.zip','w') as arc:
            for f in (src/name).rglob('*'):
                if f.is_file():arc.write(f,f'{name}/{f.name}')
    _extract_all_archives(zips,tmp_path/'unzipped')
    samples=_prepare_m3fd(zips,ADDITIONAL_DATASETS['m3fd'],0.25,0)
    assert len(samples)==6
    assert all(s.labels for s in samples)


def test_threedataset_command_handles_all_three_local_fixture_sources(tmp_path):
    # Held-out builder requires 15+ source train pairs; use 24+ pairs.
    sources={
        'vedai':create_vedai(tmp_path/'src_vedai',pairs=32),
        'm3fd':create_m3fd(tmp_path/'src_m3fd',pairs=32),
        'dronevehicle':create_dronevehicle(tmp_path/'src_drone',pairs=30),
    }
    result=run(output=tmp_path/'output', accept_license=True, archives=sources,
               validation_fingerprint='none')
    obj=json.loads(result.read_text())
    assert len(obj['datasets'])==3
    for record in obj['datasets']:
        assert record['validated']
        assert all(record['splits'][split]>0 for split in ('train','val','test'))
        assert Path(record['ready_yaml']).is_file()
        assert record['paper_doi'].startswith('10.')


def test_hf_archives_are_cached_without_redownloading(tmp_path, monkeypatch):
    import sys, types
    module=types.ModuleType('huggingface_hub')
    calls=[]
    def download(repo_id,filename,repo_type):
        calls.append(filename)
        src=tmp_path/'cache'/filename;src.parent.mkdir(exist_ok=True)
        src.write_bytes(b'PK\x03\x04DATA');return str(src)
    module.hf_hub_download=download
    monkeypatch.setitem(sys.modules,'huggingface_hub',module)
    _download_hf_zip_files('McCheng/DroneVehicle',('train.zip','val.zip'),tmp_path/'destination')
    _download_hf_zip_files('McCheng/DroneVehicle',('train.zip','val.zip'),tmp_path/'destination')
    assert calls == ['train.zip','val.zip']
