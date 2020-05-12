from distutils.core import setup

setup(
    name='spagat',
    version='0.0.1',
    author='Robin Beer',
    url='http://www.fz-juelich.de/iek/iek-3/EN/Home/home_node.html',
    packages=["spagat"],
    install_requires=['xarray', 'numpy', 'pandas', 'geopandas', 'dask', 'scipy']
)
