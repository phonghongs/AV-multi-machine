import logging

FORMAT = "%(asctime)-15s : %(levelname)s : %(name)s - %(message)s"
logging.basicConfig(filename="runtime.log", filemode='w', format=FORMAT)