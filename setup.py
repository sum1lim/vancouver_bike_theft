from setuptools import setup

__version__ = (0, 0, 0)

setup(
    name="vancouver_bike_theft",
    description="Data Mining for Vancouver bike theft risk prediction/assessment model",
    version=".".join(str(d) for d in __version__),
    author="Sangwon Lim",
    author_email="sangwonl@uvic.ca",
    packages=["vancouver_bike_theft"],
    include_package_data=True,
    scripts="""
        ./scripts/normalize
        ./scripts/tr_te_split
        ./scripts/NN_regression
        ./scripts/test_regression
        ./scripts/predict
    """.split(),
)
