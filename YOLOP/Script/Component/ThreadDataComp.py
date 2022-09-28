from queue import Queue
import string
import threading
from dataclasses import dataclass


@dataclass
class ThreadDataComp():
    ImageQueue: Queue
    TransformQueue: Queue
    OutputQueue: Queue
    ImageCondition: threading.Condition
    TransformCondition: threading.Condition
    OutputCondition: threading.Condition
    ImagePath: string
    ModelPath: string
    isQuit: bool
    totalTime: Queue