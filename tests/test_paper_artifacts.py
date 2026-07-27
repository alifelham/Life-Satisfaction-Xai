from hashlib import sha256
from pathlib import Path


def test_paper_notebook_checksum():
    path = Path("notebooks/paper/published_experiment.ipynb")
    expected = "5e6aa8a112239fdce406f376c8ff5f82cd5a8e3cb8a08d66b5128ee75bee29fb"
    assert sha256(path.read_bytes()).hexdigest() == expected
