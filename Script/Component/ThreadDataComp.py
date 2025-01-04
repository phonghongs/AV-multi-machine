from queue import Queue
import string
import threading
import asyncio
from dataclasses import dataclass

@dataclass
class ThreadDataComp():
    ImageQueue: Queue
    TransformQueue: Queue
    QuantaQueue: Queue
    totalTime: Queue
    ImageCondition: threading.Condition
    TransformCondition: threading.Condition
    QuantaCondition: threading.Condition
    OutputCondition: threading.Lock
    ImagePath: string
    ModelPath: string
    isQuit: bool
    isTimeProcess: bool
    output : list
