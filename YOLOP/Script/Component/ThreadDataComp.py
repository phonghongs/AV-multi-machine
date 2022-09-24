from queue import Queue
import string
from dataclasses import dataclass

@dataclass
class ThreadDataComp():
    ImageQueue: Queue
    TransformQueue: Queue
    OutputQueue: Queue
    ImagePath: string
    ModelPath: string
    isQuit: bool