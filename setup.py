# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from setuptools import find_packages, setup

python_requires = ">=3.10.14,<3.12.0"

if __name__ == "__main__":
    setup(
        name="z0inf",
        version=0.1,
        description="Zero-order approximation of influence functions",
        license="Apache-2.0",
        packages=find_packages(
            exclude=[
                "examples",
                "tests",
            ]
        ),
        install_requires=[
            'torch>=2.4.1',
        ],
        python_requires=python_requires,
    )