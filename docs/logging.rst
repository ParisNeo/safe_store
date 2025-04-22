=======
Logging
=======

``safe_store`` uses the `ascii_colors <https://github.com/ParisNeo/ascii_colors>`_ library for internal logging, providing clear, leveled, and colorful console output by default.

Default Behavior
----------------

*   Logs are printed directly to the console (stderr).
*   Messages are color-coded based on severity (DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL).
*   The default logging level is ``INFO``. This means only messages with severity INFO, SUCCESS, WARNING, ERROR, and CRITICAL will be displayed. DEBUG messages are hidden.

Changing the Log Level
----------------------

You can easily change the minimum severity level displayed when initializing ``safe_store``:

.. code-block:: python

   import safe_store
   from safe_store import LogLevel # Or from ascii_colors import LogLevel

   # Show only warnings and errors
   store_warn = safe_store.safe_store("my_store_warn.db", log_level=LogLevel.WARNING)

   # Show all messages, including detailed debug info
   store_debug = safe_store.safe_store("my_store_debug.db", log_level=LogLevel.DEBUG)

Advanced Configuration (Global)
-------------------------------

Since ``safe_store`` uses ``ascii_colors``, you can configure logging globally for your entire application *before* initializing ``safe_store``. This allows you to:

*   Log messages to a file.
*   Change the output format.
*   Use JSON formatting.
*   Add multiple handlers (e.g., log DEBUG to file, INFO to console).
*   Disable console logging entirely.

Here's how to configure ``ascii_colors`` globally:

.. code-block:: python

   import safe_store
   from ascii_colors import ASCIIColors, LogLevel, FileHandler, Formatter, JSONFormatter
   import logging # Standard logging Formatter can also be used

   # --- Example 1: Set global level and log to file ---
   ASCIIColors.set_log_level(LogLevel.DEBUG) # Apply DEBUG level globally

   # Create a file handler
   log_file = "app_activity.log"
   file_handler = FileHandler(log_file, encoding='utf-8')

   # Set a specific format for the file
   file_formatter = Formatter(
       "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
       datefmt="%Y-%m-%d %H:%M:%S"
   )
   file_handler.setFormatter(file_formatter)

   # Add the file handler to ascii_colors
   ASCIIColors.add_handler(file_handler)

   # Optional: If you ONLY want file logging, remove the default console handler
   # default_console_handler = ASCIIColors.get_default_handler()
   # if default_console_handler:
   #    ASCIIColors.remove_handler(default_console_handler)

   print(f"Logging DEBUG and above to console (default) and {log_file}")

   # Now initialize safe_store - it will respect the global settings
   store = safe_store.safe_store("my_store.db")
   # ... use store ...
   # safe_store's internal DEBUG messages will now appear in the file


   # --- Example 2: JSON logging to file ---
   # Clear previous handlers if starting fresh configuration
   # ASCIIColors.reset() # Or ASCIIColors.clear_handlers()

   # ASCIIColors.set_log_level(LogLevel.INFO) # Set desired level

   # json_handler = FileHandler("app_log.jsonl", encoding='utf-8')
   # json_formatter = JSONFormatter()
   # json_handler.setFormatter(json_formatter)
   # ASCIIColors.add_handler(json_handler)

   # # Optionally remove console handler
   # # default_console_handler = ASCIIColors.get_default_handler()
   # # if default_console_handler: ASCIIColors.remove_handler(default_console_handler)

   # store_json = safe_store.safe_store("my_json_store.db")
   # ... use store_json ...

See the `ascii_colors documentation <https://github.com/ParisNeo/ascii_colors#usage>`_ for more details on handlers and formatters.
