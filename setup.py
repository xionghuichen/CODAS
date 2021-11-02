#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages

setup(
        name='codas',
        version=1.0,
        description=(
            'Sim2Real with GAIL'
        ),
        author='Xiong-Hui Chen, Shengyi Jiang, Feng Xu and Yang Yu',
        author_email='chenxh@lamda.nju.edu.cn, shengyi.jiang@outlook.com, xuf@lamda.nju.edu.cn, yangyu@lamda.nju.edu.cn',
        maintainer='Xiong-Hui Chen, Shengyi Jiang, Feng Xu and Yang Yu',
        package_data={'codas': ['mj_env/assets/*.xml']},
        packages=[package for package in find_packages(exclude=['secret/*'])
                        if package.startswith("codas")],
        platforms=["all"],
        install_requires=[
            "stable_baselines=2.10",
        ]
    )
