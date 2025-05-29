from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="adv_llm",
    version="0.1.0",
    author="Chung-en Sun, Xiaodong Liu",
    description="A tool for adversarial training of large language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "advllm_train=adv_llm.train_cli:main",
            "advllm_get_adv_prompts=adv_llm.get_adv_prompts_cli:main",
            "advllm_eval=adv_llm.eval_cli:main",
        ],
    },
) 