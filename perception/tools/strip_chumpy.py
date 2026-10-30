"""Convert the official MANO_RIGHT.pkl into a chumpy-free pickle (plain numpy + scipy.sparse).

The official MANO file stores some arrays as chumpy objects; chumpy is unmaintained, so we do not want it at runtime.
This script unpickles chumpy objects as stubs (no chumpy import needed), pulls out the numeric array each one holds,
and writes a plain pickle that smplx / WiLoR can load.

Run from the repo root (hoi-perception env):
    python perception/tools/strip_chumpy.py
Reads  third_party/WiLoR/mano_data/MANO_RIGHT_orig.pkl (the untouched official file)
Writes third_party/WiLoR/mano_data/MANO_RIGHT.pkl
"""

import pickle
import sys

import numpy as np
import scipy.sparse as sp

sys.modules["chumpy"] = None   # guard: prove this works without chumpy (any real chumpy import now fails)

MANO_DIR = "third_party/WiLoR/mano_data"


class ChumpyStub:
    """Stands in for a chumpy object; ``chumpy_class`` names the original class, ``state`` holds its pickled state dict."""
    chumpy_class = "?"

    def __setstate__(self, state):
        self.state = state


class NoChumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return type(f"Stub_{name}", (ChumpyStub,), {"chumpy_class": f"{module}.{name}"})
        return super().find_class(module, name)


def numeric_arrays(state):
    return {k: v for k, v in state.items() if isinstance(v, np.ndarray) and v.dtype.kind in "fiu"}


def resolve(stub):
    """Value of a chumpy object. Only two classes occur in MANO: a plain Ch (holds the array in ``x``) and a Select
    (a view: ``a.ravel()[idxs].reshape(preferred_shape)``). Anything else is an error, not a guess."""
    name = stub.chumpy_class.rsplit(".", 1)[-1]
    st = stub.state
    if name == "Ch":
        extra = sorted(set(numeric_arrays(st)) - {"x"})
        if extra:
            raise ValueError(f"{stub.chumpy_class}: unexpected numeric state {extra} (a computed op, not a plain Ch)")
        return np.asarray(st["x"])
    if name == "Select":
        return resolve(st["a"]).ravel()[st["idxs"]].reshape(st["preferred_shape"])
    raise ValueError(f"unsupported chumpy class {stub.chumpy_class} (state keys {sorted(st)})")


EXPECTED_SHAPES = {"v_template": (778, 3), "shapedirs": (778, 3, 10), "posedirs": (778, 3, 135), "J_regressor": (16, 778),
                   "weights": (778, 16), "f": (1538, 3), "hands_components": (45, 45), "hands_mean": (45,), "kintree_table": (2, 16)}
INTEGER_KEYS = {"f", "kintree_table"}


def check(clean):
    assert clean["shapedirs"].shape == (778, 3, 10) and clean["shapedirs"].dtype == np.float64, "shapedirs must be (778,3,10) float64"
    for k, shape in EXPECTED_SHAPES.items():
        assert clean[k].shape == shape, f"{k}: shape {clean[k].shape}, expected {shape}"
    for k, v in clean.items():
        if isinstance(v, np.ndarray) and v.dtype.kind in "iu":
            assert k in INTEGER_KEYS, f"{k}: unexpected integer array"


def main():
    with open(f"{MANO_DIR}/MANO_RIGHT_orig.pkl", "rb") as f:
        data = NoChumpyUnpickler(f, encoding="latin1").load()
    clean = {}
    for k, v in data.items():
        if isinstance(v, ChumpyStub):
            print(f"{k}: {v.chumpy_class}, state keys {sorted(v.state)}")
            clean[k] = resolve(v)
        else:
            clean[k] = v
    check(clean)
    with open(f"{MANO_DIR}/MANO_RIGHT.pkl", "wb") as f:
        pickle.dump(clean, f, protocol=4)
    for k, v in clean.items():
        shape = v.shape if (isinstance(v, np.ndarray) or sp.issparse(v)) else ()
        print(f"{k:<18} {type(v).__module__}.{type(v).__name__:<12} {shape}")


if __name__ == "__main__":
    main()
