if [[ -z "${1:-}" ]]; then
    rm -rf ./build ./zipper/*.so
    python setup.py build_ext --inplace 2>&1 | tee build.txt
    python -c 'import torch; import zipper._C as C; print(C);  print("ops names:", [n for n in dir(C.ops) if not n.startswith("_")])'
    python -m pip install -e . --no-build-isolation
elif [[ "$1" == "build" ]]; then
    rm -rf ./build ./zipper/*.so
    python setup.py build_ext --inplace 2>&1 | tee build.txt
    python -c 'import torch; import zipper._C as C; print(C);  print("ops names:", [n for n in dir(C.ops) if not n.startswith("_")])'
elif [[ "$1" == "install" ]]; then
    python -m pip install -e . --no-build-isolation
else
    echo "Unknown argument: $1"
    exit 1
fi
