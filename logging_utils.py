import logging

# Configure logging
logging.basicConfig(
    filename="logs/application.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_to_ui_and_file(ui_logger, message):
    logging.info(message)
    ui_logger.log_message(message)