# Configuration

SynthDoc is designed for easy configuration using a `.env` file. This allows you to securely store your API keys and define your preferred LLM models without hardcoding them into your scripts.

## Using a `.env` File

This is the recommended way to configure SynthDoc.

1.  **Create a `.env` file**:
    In the root of your project, create a file named `.env`. You can do this by copying the provided template:
    ```bash
    cp env.template .env
    ```

2.  **Add your API keys**:
    Open the `.env` file and add the API keys for the LLM providers you want to use. You can also set a default model.

    ```env
    # .env example
    OPENAI_API_KEY=sk-your-openai-api-key
    GROQ_API_KEY=gsk_your_groq_api-key
    ANTHROPIC_API_KEY=sk-ant-your-anthropic-key

    # Set the default model to use for generation
    DEFAULT_LLM_MODEL=gpt-4o-mini
    ```

SynthDoc will automatically load these variables when you initialize the `SynthDoc` class.

## Manual Configuration

If you prefer not to use a `.env` file, you can pass your configuration directly to the `SynthDoc` constructor.

### Overriding the Model and API Key

```python
from synthdoc import SynthDoc

# Manually specify the model and API key
synth = SynthDoc(
    llm_model="groq/llama-3-8b-8192",
    api_key="your-groq-key"
)
```

### Using Local Models with Ollama

If you are running models locally with Ollama, you can specify the model name directly. No API key is needed.

```python
from synthdoc import SynthDoc

# Connect to a local Ollama model
synth = SynthDoc(llm_model="ollama/llama2")
```
