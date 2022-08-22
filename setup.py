import numpy as np
from setuptools import Extension, setup


def main():
    setup(
        name="astar",
        version="1.0.0",
        description="Python interface for the fputs C library function",
        author="jack",
        author_email="jack@gmail.com",
        ext_modules=[
            Extension(
                "astar", ["hades/extensions/astar.c"], include_dirs=[np.get_include()]
            )
        ],
    )


if __name__ == "__main__":
    main()
