# ![SPHinXsys Logo](assets/logo.png) SPHinXsys

**Notice on repository transfer to SPHinXsys team** 

In order to promoting open-source democratization,
this repository will be transferred to the SPHinXsys team 
in the next few weeks.
After the transfer, the decision-making process for SPHinXsys will be made by a number of project leaders from different institutions.

**Project Status**  
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Linux](https://img.shields.io/badge/os-Linux-green.svg)](https://shields.io/)
[![Windows](https://img.shields.io/badge/os-Windows-green.svg)](https://shields.io/)
[![macOS](https://img.shields.io/badge/os-macOs-green.svg)](https://shields.io/)
![ci workflow](https://github.com/Xiangyu-Hu/SPHinXsys/actions/workflows/ci.yml/badge.svg?event=push)

**Project Communication**  
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/sphinxsys.svg?style=social&label=Follow%20%40sphinxsys)](https://twitter.com/sphinxsys)
[![YouTube](https://img.shields.io/badge/YouTube-FF0000.svg?style=flat&logo=YouTube&logoColor=white)](https://www.youtube.com/channel/UCexdJbxOn9dvim6Jg1dnCFQ)
[![Bilibili](https://img.shields.io/badge/bilibili-%E5%93%94%E5%93%A9%E5%93%94%E5%93%A9-critical)](https://space.bilibili.com/1761273682/video)

## Description

SPHinXsys (pronunciation: s'fink-sis) is an acronym from **S**moothed **P**article **H**ydrodynamics for **in**dustrial comple**X** **sys**tems.
The multi-physics library uses SPH (smoothed particle hydrodynamics) as the underlying numerical method
for both particle-based and mesh-based discretization.
Due to the unified computational framework, SPHinXsys is able to carry out simulation and optimization at the same time.
For more information on the SPHinXsys project, please check the project website: <https://www.sphinxsys.org>.

## Examples at a glance

Using SPHinXsys library, straightforward and fast multi-physics modeling can be achieved.
Here, we present several short examples in flow, solid dynamics, fluid structure interactions (FSI) and dynamic solid contact.

<a href="https://github.com/Xiangyu-Hu/SPHinXsys/blob/master/tests/2d_examples/test_2d_dambreak/Dambreak.cpp">
<img src="https://github.com/Xiangyu-Hu/SPHinXsys-public-files/blob/master/videos/dambreak.gif" height="192px"></a>
<a href="https://github.com/Xiangyu-Hu/SPHinXsys/blob/master/tests/2d_examples/test_2d_fsi2/fsi2.cpp">
<img src="https://github.com/Xiangyu-Hu/SPHinXsys-public-files/blob/master/videos/fsi-2d.gif" height="192px"></a>
<a href="https://github.com/Xiangyu-Hu/SPHinXsys/blob/master/tests/3d_examples/test_3d_elasticSolid_shell_collision/3d_elasticSolid_shell_collision.cpp">
<img src="https://github.com/Xiangyu-Hu/SPHinXsys-public-files/blob/master/videos/ball-shell.gif" height="192px"></a>
<a href="https://github.com/Xiangyu-Hu/SPHinXsys/blob/master/tests/3d_examples/test_3d_shell_stability_half_sphere/test_3d_shell_stability_half_sphere.cpp">
<img src="https://github.com/Xiangyu-Hu/SPHinXsys-public-files/blob/master/videos/half-sphere.gif" height="192px"></a>
<a href="https://github.com/Xiangyu-Hu/SPHinXsys/blob/master/tests/3d_examples/test_3d_twisting_column/twisting_column.cpp">
<img src="https://github.com/Xiangyu-Hu/SPHinXsys-public-files/blob/master/videos/twisting.gif" height="168px"></a>
<a href="https://github.com/Xiangyu-Hu/SPHinXsys/blob/master/tests/2d_examples/test_2d_flow_stream_around_fish/2d_flow_stream_around_fish.cpp">
<img src="https://github.com/Xiangyu-Hu/SPHinXsys-public-files/blob/master/videos/fish-swimming.gif" height="168px"></a>
<a href="https://github.com/Xiangyu-Hu/SPHinXsys/blob/master/tests/2d_examples/test_2d_column_collapse/column_collapse.cpp">
<img src="https://github.com/Xiangyu-Hu/SPHinXsys-public-files/blob/master/videos/2d_column_collapse.gif" height="168px"></a>

## Fully compatible to classical FVM method

Through the unified computational framework in SPHinXsys,
the algorithms for particle methods are full compatible to those in the classical finite volume method (FVM).
The following gives an example of the flow around cylinder problem solved by FVM in SPHinXsys.

<a href="https://github.com/Xiangyu-Hu/SPHinXsys/blob/master/tests/2d_examples/test_2d_FVM_flow_around_cylinder/2d_FVM_flow_around_cylinder.cpp">
<img src="https://github.com/Xiangyu-Hu/SPHinXsys-public-files/blob/master/videos/fvm-sphinxsys-flow-around-cylinder.gif" height="168px"></a>

Note that the code for FVM algorithm is exact the same one for particle interaction in SPHinXsys.
The only difference is that SPHinXsys reads a predefined mesh, other than generate particles, before the computation.

## Target-driven optimization

The unique target-driven optimization is able to achieve the optimization target and physical solution all-in-once, 
which is able to accelerate optimization process greatly.
The following gives an example of optimizing the conductivity distribution 
for a thermal domain problem targeting minimum average temperature.

<a href="https://github.com/Xiangyu-Hu/SPHinXsys/blob/master/tests/optimization/test_2d_VP_heat_flux_optimization/VP_heat_flux_optimization.cpp">
<img src="https://github.com/Xiangyu-Hu/SPHinXsys-public-files/blob/master/videos/optimization.gif" height="192px"></a>

Note that the physical solution of the thermal domain (right) and the optimal distribution of conductivity (left)
are obtained at the same time when optimization is finished. 
Also note that the entire optimization process is very fast and 
only several times slower than that for a single physical solution with given conductivity distribution.  

## Python interface

While SPHinXsys is written in C++, it provides a python interface for users to write python scripts to control the simulation, 
including carry out regression tests for continuous integration (CI) and other tasks.
One example is given below for the dambreak case.
Please check the source code of 
[2D Dambreak case with python interface](https://github.com/Xiangyu-Hu/SPHinXsys/tree/master/tests/2d_examples/test_2d_dambreak_python) 
for the usage.

## Heterogenous computing

Recently, we have a preview release for the heterogeneous computing version of SPHinXsys. 
By using SYCL, a royalty-free open standard developed by the Khronos Group that allows developers
to program heterogeneous architectures in standard C++, SPHinXsys is able to utilize the power of GPU.
Please check the [Preview Release](https://github.com/Xiangyu-Hu/SPHinXsys/releases/tag/v1.0-beta.08-sycl)
and the [SYCL branch](https://github.com/Xiangyu-Hu/SPHinXsys/tree/sycl) for details.

## Publications

Main publication on the library:

1. C. Zhang, M. Rezavand, Y. Zhu, Y. Yu, D. Wu, W. Zhang, J. Wang, X. Hu,
"SPHinXsys: an open-source multi-physics and multi-resolution library based on smoothed particle hydrodynamics",
Computer Physics Communications, 267, 108066, 2021.  
[![Main Publication](https://img.shields.io/badge/doi-10.1016%2Fj.cpc.2021.108066-d45815.svg)](https://doi.org/10.1016/j.cpc.2021.108066)

The numerical methods and computational algorithms in SPHinXsys are based on the following [publications](assets/publication.md).

## Software Architecture

SPHinXsys is cross-platform can be compiled and used in Windows, Linux and McOS systems.

## Installation, tutorial and documentation

For installation, program manual and tutorials, please check <https://www.sphinxsys.org/html/sphinx_index.html>.
Please check the documentation of the code at <https://xiangyu-hu.github.io/SPHinXsys/>.
For a Docker image, check <https://hub.docker.com/r/toshev/sphinxsys>.

## Get involved to SPHinXsys

Thank you for using and supporting our open-source project! We value all feedback and strive to improve our codebase continuously.

As the code is on git-hub, you can register an account there (if you do not have a github account yet)
and fork out the SPHinXsys repository.
You can work on the forked repository and add new features, and then commit them.
You can also initiate a pull request to the main repository,
so that your new features can be merged into it.

To ensure efficient and effective development, we prioritize addressing issues and pull requests from those who actively contribute to the project. Your contributions, whether through code, documentation, or other means, help us maintain and enhance the project for everyone.
We encourage all users to consider contributing in any way they can. Together, we can build a better, more robust software.


You are also welcomed to join the main repository as a collaborator,
by which you are able to branch directly in the main repository,
and review the pull request.

If you have any further question, you are also welcomed to contact <xiangyu.hu@tum.de>.
