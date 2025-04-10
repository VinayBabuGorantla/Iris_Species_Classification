import sys
from src.logger import logging

def error_message_details(error, error_detail: sys):
    """
    This function generates a detailed error message including the filename, line number, and error message.
    """
    _, _, exc_tb = error_detail.exc_info()  # Extracts the traceback details
    file_name = exc_tb.tb_frame.f_code.co_filename  # File name where the error occurred
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error))  # Formatting the error message
    return error_message


class CustomException(Exception):
    """
    A custom exception class that captures the error message and traceback details
    to provide better logging and debugging information.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail=error_detail)  # Generate detailed error message
    
    def __str__(self):
        return self.error_message  # Return the detailed error message when the exception is printed
