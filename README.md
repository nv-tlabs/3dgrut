# Ray Tracing Gaussian Splats


## Using the GUI

To use the interactive UI, install the following optional dependencies, and run with `--with-gui`

```
python -m pip install git+ssh://git@github.com/nmwsharp/polyscope-py.git@v2
python -m pip install cuda-python cupy
```

The latter two packages `cuda-python` and `cupy` may very slow to install and/or create CUDA versioning problems, which is why we don't install them by default. (In the future we could remove these by writing a few of our own cuda bindings.)
