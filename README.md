# BLAB Chatbot - DEEPAGÉ

This Python module allows the integration of [DEEPAGÉ](../../../deepage) with
[BLAB](../../../blab-controller).

**Currently, the Python versions supported by this module are {_v_ | 3.8.1 ≤ _v_ < 3.9}.**
<!-- At the time of writing, apparently the latest Haystack version requires
     an old version of Elasticsearch, which does not support Python 3.10 -->

### Installation

- Obtain the trained DEEPAGÉ model.
  It should be a directory containing files such as *pytorch_model.bin* and *optimizer.pt*.
  If you don't have access to the model, please contact the authors of DEEPAGÉ.

- Obtain the document containing the texts.
  Each line should have a title and a text, separated by a tab character.
  If you don't have access to the documents, please contact the authors of DEEPAGÉ.

- Install a version of
  [Python](https://www.python.org/downloads/release/python-3816/) between 3.8.1 (inclusive)
  and 3.9 (exclusive).
  Other versions are not supported.

- Make sure that [distutils](https://docs.python.org/3/library/distutils.html) is installed
  (e.g. the package `python3.8-distutils` in some Linux distributions). It should be possible
  to execute `python3.8 -c 'import distutils; print("OK")'` without errors.

- Install Elasticsearch 7.10 - see instruction to install
  [from a .tar.gz archive](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/targz.html)
  or [as a deamon](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/deb.html).
  Other versions may not be supported.

- Follow [these installation instructions](../../../blab-chatbot-bot-client/blob/main/INSTALL.md)
  using [*settings_deepage_template.py*](settings_deepage_TEMPLATE.py) as a template.

  If errors are shown, install the following packages on your system and try again:

  ```
  libblas3 liblapack3 liblapack-dev libblas-dev gfortran libatlas-base-dev
  ```

- Start Elasticsearch and wait a few seconds.

- Enter the *src/* directory and create the index:

  ```shell
  poetry run ./run.py --config=name_of_your_config_file.py index
  ```

- Follow [these instructions](../../../blab-chatbot-bot-client/blob/main/RUN.md) to execute the
  program. The addition to your controller settings can be:

  ```python
  CHAT_INSTALLED_BOTS.update({
      "DEEPAGÉ": websocket_external_bot(url="http://localhost:25226"),
  })
  ```
