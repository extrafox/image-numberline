"""Download a random sample of YFCC100M images from the public multimedia-commons bucket."""
import random, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET = "s3://multimedia-commons/data/images/"
out = Path(sys.argv[1]); n_top = int(sys.argv[2]); leaves_per_top = int(sys.argv[3])
rng = random.Random(0)

def aws(*args):
    return subprocess.run(["aws", "s3", *args, "--no-sign-request"] + (["--only-show-errors"] if args[0] == "cp" else []),
                          capture_output=True, text=True).stdout

def fetch(top):
    subs = [l.split()[-1] for l in aws("ls", BUCKET + top + "/").splitlines() if "PRE" in l]
    for sub in rng.sample(subs, min(leaves_per_top, len(subs))):
        aws("cp", "--recursive", BUCKET + top + "/" + sub, str(out / top / sub))

tops = rng.sample([f"{i:03x}" for i in range(4096)], n_top)
with ThreadPoolExecutor(32) as ex:
    list(ex.map(fetch, tops))
