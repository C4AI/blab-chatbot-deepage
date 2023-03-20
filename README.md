# BLAB Chatbot - DEEPAGÉ

This Python module allows the integration of [DEEPAGÉ](../../../deepage) with
[BLAB](../../../blab-controller).

**Currently, the Python versions supported by this module are {_v_ | 3.8.1 ≤ _v_ < 3.9}.**
<!-- At the time of writing, apparently the latest Haystack version requires
     an old version of Elasticsearch, which does not support Python 3.10 -->

### Installation


- Obtain the trained DEEPAGÉ model.
  It should be a directory containing files such as *pytorch_model.bin* and *optimizer.pt*.

- Obtain the document containing the texts.
  Each line should have a title and a text, separated by a tab character.

- Install a version of
  [Python](https://www.python.org/downloads/release/python-3816/) between 3.8.1 (inclusive)
  and 3.9 (exclusive).
  Other versions are not supported.

- Make sure that [distutils](https://docs.python.org/3/library/distutils.html) is installed 
  (e.g. the package `python3.8-distutils` in some Linux distributions). It should be possible
  to execute `python3.8 -c 'import distutils; print("OK")'` without errors.

- Install [Poetry](https://python-poetry.org/) (version 1.2 or newer):

  ```shell
  curl -sSL https://install.python-poetry.org | python3 -
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

- Install Elasticsearch 7.10 - see instruction to install
  [from a .tar.gz archive](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/targz.html)
  or [as a deamon](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/deb.html).
  Other versions may not be supported.

- Start Elasticsearch.

- Optionally, run `poetry shell` to open a shell that uses the virtual environment, and
  all the commands below can be executed on that shell without prefixing them with `poetry run`.

- Enter the *src/* directory and create the index:

  ```shell
  poetry run python -m blab_chatbot_deepage index
  ```

- In order to open an interactive demo that answers questions, run:

  ```shell
  poetry run python -m blab_chatbot_deepage answer
  ```

- In order to start the server that will interact with BLAB Controller, run:

  ```shell
  poetry run python -m blab_chatbot_deepage startserver
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
