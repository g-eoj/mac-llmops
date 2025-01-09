# mac-llmops

The new Apple silicon is great for running LLMs.
This library attempts to simplify the process of trying new LLMs by:

- simplifying local model cache management (despite the extra memory Apple gives us now, the default SSD size is still painfully small)
- providing local and online model searches that automatically filter out models that have no chance of fitting in your Mac's GPU memory

## Usage

See the example notebook.
