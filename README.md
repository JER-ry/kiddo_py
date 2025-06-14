# kiddo_py

Python bindings for the [Kiddo](https://crates.io/crates/kiddo) k-d tree library using [PyO3](https://pyo3.rs/), focusing on [`ImmutableKdTree`'s `within_unsorted()`](https://docs.rs/kiddo/latest/kiddo/immutable/float/kdtree/struct.ImmutableKdTree.html#method.within_unsorted). Uses [Rayon](https://crates.io/crates/rayon) for parallelism.

> Code generated with LLM, with manual edits.

## Installation

```bash
# prepare
git clone https://github.com/JER-ry/kiddo_py.git --depth=1
cd kiddo_py
pip install maturin

# install in development mode
maturin develop
# or build a wheel
maturin build --release
```

## Usage

See `examples.py`.

Only 2D and 3D are supported, but you can check `src/lib.rs` and easily add support for any dimension.

Note that `.astype(np.float32)` or `dtype=np.float32` is required. Refer to [kiddo's documentation](https://docs.rs/kiddo/latest/kiddo/float/kdtree/struct.KdTree.html) for how to use `f64`.