# BLAB Chatbot - DEEPAGÉ

This Python module allows the integration of [DEEPAGÉ](../../../deepage) with
[BLAB](../../../blab-controller).

**Currently, only Python 3.8 is supported.**
<!-- At the time of writing, apparently the latest Haystack version requires
     an old version of Elasticsearch, which does not support Python 3.10 -->

### Installation

- Obtain the trained DEEPAGÉ model.
  It should be a directory containing files such as *pytorch_model.bin* and *optimizer.pt*.

- Obtain the document containing the texts.
  Each line should have a title and a text, separated by a tab character.

- Install Elasticsearch 7.10 - [see instructions](https://www.elastic.co/guide/en/elasticsearch/reference/current/targz.html).

- Start Elasticsearch.

- Create a file named *settings.ini* in the same directory as this *README.md* file and add the required fields as follows:
  ```ini
  [blab_chatboot_deepage]
  index_name=name_of_the_index
  document=data/document.csv
  model=model/your-model-directory
  server_host=127.0.0.1
  server_port=25226
  ws_url=ws://localhost:8000

  ```
  Note that non-absolute paths will be relative to this directory.
  Index name should contain only lowercase letters and underscores.
  Server host should be `127.0.0.1` to accept only local connections from the controller,
  and the port is arbitrary as long as it is available.
  The WebSocket URL is the controller address and must start with `ws://` or `wss://`
  (the same path used by the frontend).

- Open your controller settings file (`dev.py` or `prod.py`) and update
  the `CHAT_INSTALLED_BOTS` dictionary to include the DEEPAGÉ settings.
  Example:

```python
CHAT_INSTALLED_BOTS.update(
    {
        'DEEPAGÉ': (
            'chat.bots',
            'WebSocketExternalBot',
            ['http://localhost:25226/'],
            {},
        ),
    }
)

```
