from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    py_modules=['click_app'],
    install_requires=[
        'click',
    ],
    entry_points={
        'console_scripts':[
            'from_wav = click_app:wav_to_transcript',
            'from_url = click_app:url_to_transcript',
        ],
    },
    description='Speech-to-text-experiment',
    author='Dagmar Schoenenberg',
)
