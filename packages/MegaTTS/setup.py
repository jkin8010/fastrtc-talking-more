from setuptools import setup, find_packages

setup(
    name="megatts3",
    version="0.0.0",
    description="MegaTTS Text-to-Speech Model",
    url="https://github.com/bytedance/MegaTTS3",
    packages=find_packages(include=["megatts3", "megatts3.*", "MegaTTS3", "MegaTTS3.*"]),
    license="Apache-2.0 license",
    install_requires=[
        "torch>=2.2.1",
        "numpy>=2.0.2",
        "pydub>=0.25.1",
        "pyloudnorm>=0.1.1",
        "langdetect>=1.0.9",
        "librosa>=0.11.0",
        "transformers>=4.0.0",
    ],
    platforms="any",
    include_package_data=True,  # 启用包含非代码文件
    package_data={
        "megatts3": ["**/*.json", "**/*.yaml", "**/*.npy"],  # 指定要包含的 JSON 文件
    },
)