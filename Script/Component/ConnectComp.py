from http import server
import string
from dataclasses import dataclass

@dataclass
class ConnectComp():
    serverIP : string
    serverPort : string
    readyToStart : bool