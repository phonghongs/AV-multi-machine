from queue import Queue
import string
import threading
import asyncio
from dataclasses import dataclass


@dataclass
class ThreadDataComp():
    ImageQueue: Queue
    TransformQueue: Queue
    OutputQueue: asyncio.Queue
    ImageCondition: threading.Condition
    TransformCondition: threading.Condition
    OutputCondition: threading.Lock
    ImagePath: string
    ModelPath: string
    isQuit: bool
    totalTime: Queue
    output : list
    QuantaQueue: Queue
    QuantaCondition: threading.Condition