# Installation

You can install SynthDoc using pip. For the best experience, we recommend installing it with the `[llm]` extra, which includes support for Large Language Models.

## Recommended Installation

This installation includes `litellm` to connect to over 100 LLM providers.

```bash
pip install synthdoc[llm]
```

## Basic Installation

If you only need the core functionalities without LLM-based content generation, you can perform a basic installation.

```bash
pip install synthdoc
```

## Development Installation

If you plan to contribute to SynthDoc, you should clone the repository and install it in editable mode.

```bash
git clone https://github.com/adithya-s-k/SynthDoc.git
cd SynthDoc
pip install -e .[llm]
```
