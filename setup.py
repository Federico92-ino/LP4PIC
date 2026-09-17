from setuptools import setup, find_packages
import numpy

about = dict()
with open("lp4pic/__about__.py") as fp:
    exec(fp.read(), about)

__version__ = about['__version__']

# with open('README.md') as f:
# long_description = f.read()

requirements = ['numpy>=1.20,<2.0', 'scipy', 'scikit-image','fbpic','h5py','opencv-python==4.9.0.80']
setup(
    name='lp4pic',
    version=__version__,
    # description="",
    # long_description=long_description,
    # long_description_content_type='text/markdown',
    packages=find_packages('.'),
    author='Federico',
    author_email='federico.avella@ino.cnr.it',
    license='GNU GPLv3',
    classifiers=[
        'Programming Language :: Python',
        'Natural Language :: English',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: Python :: 3.11'],
    # ext_modules=cythonize(extensions),
    install_requires=[requirements],
    # ext_package='lampy/compiled_cython',
    # libraries=[lib_read_binary, lib_read_phase_space],
    include_dirs=[numpy.get_include()]
)