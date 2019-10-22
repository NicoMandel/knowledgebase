# Knowledge Server package

This package is sought to implement the work as written in some documentation. It is the ROS-implementation of the [numerical simulation](https://github.com/NicoMandel/Numerical-Sim-Semantic)
This package co-exists with the [aruco_client](https://github.com/NicoMandel/aruco_client) package

## Dependencies
* The [ml_detector package](https://github.com/qutas/marker_localization) from qutas github
* Px4 Firmware
* A proprietary camera model for the UAV to use (not included)
* Numpy and Pandas
* A fast-text word vector file (too large to include). Should be named according to [this line](https://github.com/NicoMandel/knowledgebase/blob/master/src/word2vec_mapping.py#L32), placed in the `config` folder and can be downloaded from [here](https://fasttext.cc/docs/en/english-vectors.html) approx. 600 MB download. 

## Known issues
* it has been attempted to move `cv_bridge` dependencies away from this package, and into the aruco_client. These 2 packages co-exist
* the camera model of px4 is not recommended.
* python2 and python3 issues - change shebangs where necessary
