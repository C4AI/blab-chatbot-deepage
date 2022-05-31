# BLAB Chatbot - DEEPAGÉ

This Python module allows the integration of [DEEPAGÉ](../../../deepage) with
BLAB.

### Installation

- Obtain the trained DEEPAGÉ model and take a note of the full path where it was saved. 

- ~~Install Elasticsearch - [see instructions](https://www.elastic.co/guide/en/elasticsearch/reference/current/targz.html).~~ **NOT USED FOR NOW**

- Open a terminal window in the directory where
  [BLAB Controller](../../../blab-controller) is installed.
- Install this module in the same environment:

  ```shell
  poetry run python -m pip install git+https://github.com/C4AI/blab-chatbot-deepage
  ```
- Open your controller settings file (`dev.py` or `prod.py`) and update
  the `CHAT_INSTALLED_BOTS` dictionary to include the DEEPAGÉ settings.
  Example:

```python
CHAT_INSTALLED_BOTS.update(
    {
        'DEEPAGÉ': (
            'blab_chatbot_deepage.deepage_bot',
            'DeepageBot',
            [],
            {
                'model_dir': '/...',  # full path to model directory
                'k_retrieval': 10,
            },
        ),
    }
)

```