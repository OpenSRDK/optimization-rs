# optimization-rs

## Usage

```toml
[dependencies]
opensrdk-optimization = "0.1.0"
blas-src = { version = "0.6", features = ["openblas"] }
lapack-src = { version = "0.6", features = ["openblas"] }
```

```rs
extern crate opensrdk_optimization;
extern crate blas_src;
extern crate lapack_src;
```

You can also use accelerate, intel-mkl, or netlib instead.
See [here](https://github.com/blas-lapack-rs/blas-lapack-rs.github.io/wiki).
