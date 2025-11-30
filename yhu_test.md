# Run the pythonbinding of the orbslam3
In the `examples` folder, run the following command to run the stereo euroc dataset.
```bash
python orbslam_stereo_euroc.py \
  /home/prism-himalayas/codespace/ORB_SLAM3/Vocabulary/ORBvoc.txt \
  /home/prism-himalayas/codespace/ORB_SLAM3/Examples/Stereo/EuRoC.yaml \
  /home/prism-himalayas/codespace/Datasets/EuRoc/MH_01_easy/mav0/cam0/data \
  /home/prism-himalayas/codespace/Datasets/EuRoc/MH_01_easy/mav0/cam1/data \
  /home/prism-himalayas/codespace/Datasets/EuRoc/MH_01_easy/mav0/cam0/data.csv
```

# Debugging
Once encounter this kind of problem:
```bash
Traceback (most recent call last):
  File "/home/prism-himalayas/codespace/ORB_SLAM-PythonBindings/examples/orbslam_stereo_euroc.py", line 7, in <module>
    import cv2
  File "/home/prism-himalayas/miniconda3/envs/orbslam3_env/lib/python3.10/site-packages/cv2/__init__.py", line 181, in <module>
    bootstrap()
  File "/home/prism-himalayas/miniconda3/envs/orbslam3_env/lib/python3.10/site-packages/cv2/__init__.py", line 153, in bootstrap
    native_module = importlib.import_module("cv2")
  File "/home/prism-himalayas/miniconda3/envs/orbslam3_env/lib/python3.10/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
ImportError: /home/prism-himalayas/miniconda3/envs/orbslam3_env/lib/python3.10/site-packages/cv2/python-3.10/../../../.././libgomp.so.1: cannot allocate memory in static TLS block
```
Using `export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1:$LD_PRELOAD` to preload the cv2 module.

When encounter this kind of problem:
```bash
Loading ORB Vocabulary. This could take a while...
Segmentation fault (core dumped)
```
Use a new package of `conda install -c conda-forge opencv-python-headless`. Remember to first delete the old 

# Logging
- [2025/11/30] Still didn't solve the problem of `Segmentation fault (core dumped)`.