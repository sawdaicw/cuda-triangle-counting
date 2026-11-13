"""Verify and unpack the original A100 result archive. No graph or GPU run."""
from pathlib import Path
import hashlib
import tarfile
import shutil

ROOT = Path(__file__).resolve().parents[1]
archive = ROOT / 'raw/a100_results_v1.tar.gz'
expected = (ROOT / 'raw/archive.sha256').read_text().split()[0]
if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
    raise ValueError('Result archive SHA-256 mismatch')
out = ROOT / 'work/input'
out.mkdir(parents=True, exist_ok=True)
with tarfile.open(archive) as stream:
    for member in stream.getmembers():
        relative = Path(member.name)
        if relative.is_absolute() or '..' in relative.parts or not (member.isfile() or member.isdir()):
            raise ValueError(f'Unsafe archive member: {member.name}')
        target = out / relative
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst)
print(f'Archive verified and unpacked to {out}')
