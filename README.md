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

- Install
  [Python 3.8](https://www.python.org/downloads/release/python-380/). Other versions may not be supported.

- Install [Poetry](https://python-poetry.org/) (version 1.2 or newer):

  ```shell
  curl -sSL https://install.python-poetry.org | python3 - --preview
  ```
  If *~/.local/bin* is not in `PATH`, add it as suggested by the output of Poetry installer.

- In the root directory of the project (which contains this _README.md_ file)
  run Poetry to install the dependencies in a new virtual environment (_.venv_):

  ```shell
  POETRY_VIRTUALENVS_IN_PROJECT=true poetry install
  ```

  If errors are shown, install the following packages on your system and try again: 
  
  ```
  libblas3 liblapack3 liblapack-dev libblas-dev gfortran libatlas-base-dev
  ```

- Create a file named *settings.ini* in the same directory as this *README.md* file and add the required fields as follows:
  ```ini
  [blab_chatbot_deepage]
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

- Install Elasticsearch 7.10 - [see instructions](https://www.elastic.co/guide/en/elasticsearch/reference/current/targz.html). Other versions may not be supported.

- Start Elasticsearch.

- Run `poetry shell` to open a sub-shell using the virtual environment. Alternatively, prefix the commands below with `poetry run `.

- Enter the *src/* directory and create the index:

  ```shell
  python -m blab_chatbot_deepage index
  ```

- In order to open an interactive demo that answers questions, run:

  ```shell
  python -m blab_chatbot_deepage answer
  ```

- In order to start the server that will interact with BLAB Controller, run:

  ```shell
  python -m blab_chatbot_deepage startserver
  ```

#### Integration with BLAB Controller

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
