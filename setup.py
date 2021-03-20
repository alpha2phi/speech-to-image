from setuptools import setup


def parse_requirements(filename):
    lines = (line.strip() for line in open(filename))
    return [line for line in lines if line and not line.startswith("#")]

setup(
    name="speech-to-image",
    version="0.1",
    description="PyTorch package for speech to image conversion.",
    url="https://github.com/alpha2phi/speech-to-image",
    author="alpha2phi",
    author_email="alpha2phi@gmail.com",
    packages=["sp2i"],
    install_requires=parse_requirements("requirements.txt"),
    zip_safe=True,
)

